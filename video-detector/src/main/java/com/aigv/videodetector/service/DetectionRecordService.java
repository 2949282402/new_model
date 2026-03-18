package com.aigv.videodetector.service;

import com.aigv.videodetector.model.DetectionRecord;
import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Stream;

@Slf4j
@Service
public class DetectionRecordService {

    private static final DateTimeFormatter FORMATTER = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    private final Map<String, DetectionRecord> records = new ConcurrentHashMap<>();

    private Path recordsFilePath;

    @Value("${records.file:}")
    private String recordsFileOverride;

    @Value("${upload.dir:}")
    private String uploadDir;

    @PostConstruct
    public void init() {
        recordsFilePath = resolveRecordsFilePath();
        log.info("History records file: {}", recordsFilePath);
        loadRecords();
        removeIncompleteRecords();
        cleanupPendingUploads();
    }

    private Path resolveRecordsFilePath() {
        if (recordsFileOverride != null && !recordsFileOverride.trim().isEmpty()) {
            return Paths.get(recordsFileOverride).toAbsolutePath().normalize();
        }
        if (uploadDir != null && !uploadDir.trim().isEmpty()) {
            Path uploadPath = resolveUploadRoot();
            Path parent = uploadPath.getParent();
            if (parent != null) {
                return parent.resolve("detection_records.txt");
            }
        }
        return Paths.get("detection_records.txt").toAbsolutePath().normalize();
    }

    private Path resolveUploadRoot() {
        return Paths.get(uploadDir).toAbsolutePath().normalize();
    }

    private void ensureRecordsFileParent() {
        if (recordsFilePath == null) {
            return;
        }
        Path parent = recordsFilePath.getParent();
        if (parent != null) {
            try {
                Files.createDirectories(parent);
            } catch (IOException e) {
                log.warn("Failed to create history record directory: {}", parent, e);
            }
        }
    }

    private void cleanupPendingUploads() {
        if (uploadDir == null || uploadDir.trim().isEmpty()) {
            return;
        }

        Path uploadPath = resolveUploadRoot();
        if (!Files.exists(uploadPath) || !Files.isDirectory(uploadPath)) {
            return;
        }

        Set<Path> keepEntries = new HashSet<>();
        for (DetectionRecord record : records.values()) {
            addKeepEntry(keepEntries, record.getFilePath());
            addKeepEntry(keepEntries, record.getFramesDir());
            addKeepEntry(keepEntries, record.getAttentionMapPath());
        }

        try (Stream<Path> entries = Files.list(uploadPath)) {
            entries.forEach(entry -> {
                Path normalized = entry.toAbsolutePath().normalize();
                if (!keepEntries.contains(normalized)) {
                    deletePathRecursively(normalized);
                }
            });
        } catch (IOException e) {
            log.warn("Failed to scan upload directory on startup: {}", uploadPath, e);
        }
    }

    private void removeIncompleteRecords() {
        boolean changed = records.entrySet().removeIf(entry -> isIncompleteRecord(entry.getValue()));
        if (changed) {
            rebuildRecordsFile();
            log.info("Removed incomplete detection records during startup cleanup");
        }
    }

    private boolean isIncompleteRecord(DetectionRecord record) {
        return isBlank(record.getFilePath())
                || isBlank(record.getResult())
                || isBlank(record.getProbability());
    }

    private boolean isBlank(String value) {
        return value == null || value.trim().isEmpty();
    }

    private void addKeepEntry(Set<Path> keepEntries, String storedPath) {
        Path entry = resolveManagedEntry(storedPath);
        if (entry != null) {
            keepEntries.add(entry);
        }
    }

    private Path resolveManagedEntry(String storedPath) {
        if (storedPath == null || storedPath.trim().isEmpty() || uploadDir == null || uploadDir.trim().isEmpty()) {
            return null;
        }
        try {
            Path uploadRoot = resolveUploadRoot();
            Path rawPath = Paths.get(storedPath);
            Path normalized = rawPath.isAbsolute()
                    ? rawPath.toAbsolutePath().normalize()
                    : uploadRoot.resolve(rawPath).toAbsolutePath().normalize();
            if (!normalized.startsWith(uploadRoot)) {
                return null;
            }
            Path relative = uploadRoot.relativize(normalized);
            if (relative.getNameCount() == 0) {
                return uploadRoot;
            }
            return uploadRoot.resolve(relative.getName(0)).normalize();
        } catch (Exception e) {
            log.warn("Failed to normalize managed path: {}", storedPath, e);
            return null;
        }
    }

    private void deletePathRecursively(Path target) {
        if (target == null || uploadDir == null || uploadDir.trim().isEmpty()) {
            return;
        }

        Path uploadRoot = resolveUploadRoot();
        Path normalized = target.toAbsolutePath().normalize();
        if (!normalized.startsWith(uploadRoot) || normalized.equals(uploadRoot) || !Files.exists(normalized)) {
            return;
        }

        try (Stream<Path> walk = Files.walk(normalized)) {
            walk.sorted(Comparator.reverseOrder()).forEach(path -> {
                try {
                    Files.deleteIfExists(path);
                } catch (IOException e) {
                    log.warn("Failed to delete stale path: {}", path, e);
                }
            });
            log.info("Removed stale unfinished content on startup: {}", normalized);
        } catch (IOException e) {
            log.warn("Failed to delete startup garbage: {}", normalized, e);
        }
    }

    public String addRecord(String originalFilename, String filePath, String framesDir, String attentionMapPath) {
        String id = UUID.randomUUID().toString();
        DetectionRecord record = new DetectionRecord(id, originalFilename, filePath, framesDir, attentionMapPath);
        records.put(id, record);
        saveRecord(record);
        log.info("Added detection record: {}, file={}", id, originalFilename);
        return id;
    }

    public void updateResult(String id, String probability, String result, String frameCount,
                             String details, Long queueDurationMs, Long detectDurationMs,
                             Integer attnTopK, Double threshold, Integer everyN) {
        DetectionRecord record = records.get(id);
        if (record == null) {
            return;
        }

        String percentageProbability = probability;
        try {
            double prob = Double.parseDouble(probability);
            percentageProbability = String.format("%.2f", prob * 100);
        } catch (NumberFormatException e) {
            log.warn("Failed to convert probability to percentage: {}", probability);
        }

        record.setProbability(percentageProbability);
        record.setResult(result);
        record.setFrameCount(frameCount);
        record.setDetails(details);
        record.setQueueDurationMs(queueDurationMs);
        record.setDetectDurationMs(detectDurationMs);
        record.setAttnTopK(attnTopK);
        record.setThreshold(threshold);
        record.setEveryN(everyN);

        rebuildRecordsFile();
        log.info("Updated detection result: {}", id);
    }

    public List<DetectionRecord> getAllRecords() {
        if (records.isEmpty()) {
            loadRecords();
        }
        List<DetectionRecord> list = new ArrayList<>(records.values());
        list.sort((a, b) -> b.getDetectTime().compareTo(a.getDetectTime()));
        return list;
    }

    public DetectionRecord getRecord(String id) {
        return records.get(id);
    }

    public boolean deleteRecord(String id) {
        DetectionRecord record = records.get(id);
        if (record == null) {
            log.warn("Record not found: {}", id);
            return false;
        }

        deleteFiles(record);
        records.remove(id);

        try {
            rebuildRecordsFile();
            return true;
        } catch (Exception e) {
            log.error("Failed to rebuild records file after delete: {}", id, e);
            return true;
        }
    }

    private void deleteFiles(DetectionRecord record) {
        Path managedEntry = resolveManagedEntry(record.getFilePath());
        if (managedEntry == null) {
            managedEntry = resolveManagedEntry(record.getFramesDir());
        }
        if (managedEntry == null) {
            managedEntry = resolveManagedEntry(record.getAttentionMapPath());
        }
        if (managedEntry == null) {
            log.warn("No managed directory found for record: {}", record.getId());
            return;
        }

        deletePathRecursively(managedEntry);
    }

    private synchronized void saveRecord(DetectionRecord record) {
        ensureRecordsFileParent();
        try (BufferedWriter writer = Files.newBufferedWriter(
                recordsFilePath,
                StandardCharsets.UTF_8,
                StandardOpenOption.CREATE,
                StandardOpenOption.APPEND)) {
            writer.write(serializeRecord(record));
            writer.newLine();
        } catch (IOException e) {
            log.error("Failed to save detection record", e);
        }
    }

    private void loadRecords() {
        records.clear();
        if (recordsFilePath == null) {
            return;
        }

        File file = recordsFilePath.toFile();
        if (!file.exists()) {
            return;
        }

        try (BufferedReader reader = Files.newBufferedReader(recordsFilePath, StandardCharsets.UTF_8)) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.trim().isEmpty()) {
                    continue;
                }
                DetectionRecord record = parseRecord(line);
                if (record != null) {
                    records.put(record.getId(), record);
                }
            }
        } catch (IOException e) {
            log.error("Failed to load history records: {}", recordsFilePath, e);
        }
    }

    private DetectionRecord parseRecord(String line) {
        String[] parts = line.split("\\|", -1);
        if (parts.length < 9) {
            log.warn("Invalid history record format: {}", line);
            return null;
        }

        try {
            DetectionRecord record = new DetectionRecord();
            record.setId(parts[0]);
            record.setOriginalFilename(parts[1]);
            record.setFilePath(parts[2]);
            record.setFramesDir(parts[3]);
            record.setAttentionMapPath(parts[4]);
            record.setProbability(parts[5].isEmpty() ? null : parts[5]);
            record.setResult(parts[6].isEmpty() ? null : parts[6]);
            record.setFrameCount(parts[7].isEmpty() ? null : parts[7]);

            if (parts.length >= 15) {
                record.setDetails(parts[8].isEmpty() ? null : decodeDetails(parts[8]));
                record.setQueueDurationMs(parts[9].isEmpty() ? null : Long.parseLong(parts[9]));
                record.setDetectDurationMs(parts[10].isEmpty() ? null : Long.parseLong(parts[10]));
                record.setAttnTopK(parts[11].isEmpty() ? null : Integer.parseInt(parts[11]));
                record.setThreshold(parts[12].isEmpty() ? null : Double.parseDouble(parts[12]));
                record.setEveryN(parts[13].isEmpty() ? null : Integer.parseInt(parts[13]));
                record.setDetectTime(LocalDateTime.parse(parts[14], FORMATTER));
            } else if (parts.length >= 12) {
                record.setDetails(parts[8].isEmpty() ? null : decodeDetails(parts[8]));
                record.setQueueDurationMs(parts[9].isEmpty() ? null : Long.parseLong(parts[9]));
                record.setDetectDurationMs(parts[10].isEmpty() ? null : Long.parseLong(parts[10]));
                record.setDetectTime(LocalDateTime.parse(parts[11], FORMATTER));
            } else if (parts.length >= 11) {
                record.setQueueDurationMs(parts[8].isEmpty() ? null : Long.parseLong(parts[8]));
                record.setDetectDurationMs(parts[9].isEmpty() ? null : Long.parseLong(parts[9]));
                record.setDetectTime(LocalDateTime.parse(parts[10], FORMATTER));
            } else {
                record.setDetectTime(LocalDateTime.parse(parts[8], FORMATTER));
            }
            return record;
        } catch (Exception e) {
            log.error("Failed to parse record line: {}", line, e);
            return null;
        }
    }

    private String serializeRecord(DetectionRecord record) {
        return String.format("%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s",
                safe(record.getId()),
                safe(record.getOriginalFilename()),
                safe(record.getFilePath()),
                safe(record.getFramesDir()),
                safe(record.getAttentionMapPath()),
                safe(record.getProbability()),
                safe(record.getResult()),
                safe(record.getFrameCount()),
                encodeDetails(record.getDetails()),
                record.getQueueDurationMs() != null ? record.getQueueDurationMs() : "",
                record.getDetectDurationMs() != null ? record.getDetectDurationMs() : "",
                record.getAttnTopK() != null ? record.getAttnTopK() : "",
                record.getThreshold() != null ? record.getThreshold() : "",
                record.getEveryN() != null ? record.getEveryN() : "",
                record.getDetectTime().format(FORMATTER));
    }

    private String safe(String value) {
        return value != null ? value : "";
    }

    private String encodeDetails(String details) {
        if (details == null || details.isEmpty()) {
            return "";
        }
        String normalized = details.replace("\r\n", "\n").replace("\r", "\n");
        return Base64.getEncoder().encodeToString(normalized.getBytes(StandardCharsets.UTF_8));
    }

    private String decodeDetails(String encoded) {
        if (encoded == null || encoded.isEmpty()) {
            return null;
        }
        try {
            return new String(Base64.getDecoder().decode(encoded), StandardCharsets.UTF_8);
        } catch (IllegalArgumentException e) {
            return encoded.replace("\\r", "\r").replace("\\n", "\n");
        }
    }

    private synchronized void rebuildRecordsFile() {
        ensureRecordsFileParent();
        try (BufferedWriter writer = Files.newBufferedWriter(
                recordsFilePath,
                StandardCharsets.UTF_8,
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING,
                StandardOpenOption.WRITE)) {
            for (DetectionRecord record : getSortedRecordsSnapshot()) {
                writer.write(serializeRecord(record));
                writer.newLine();
            }
        } catch (IOException e) {
            log.error("Failed to rebuild records file", e);
        }
    }

    private List<DetectionRecord> getSortedRecordsSnapshot() {
        List<DetectionRecord> list = new ArrayList<>(records.values());
        list.sort(Comparator
                .comparing(DetectionRecord::getDetectTime, Comparator.nullsLast(Comparator.reverseOrder()))
                .thenComparing(DetectionRecord::getId, Comparator.nullsLast(String::compareTo)));
        return list;
    }
}
