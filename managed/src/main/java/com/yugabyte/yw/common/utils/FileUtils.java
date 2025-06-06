package com.yugabyte.yw.common.utils;

import static java.nio.file.StandardCopyOption.REPLACE_EXISTING;
import static play.mvc.Http.Status.INTERNAL_SERVER_ERROR;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.yugabyte.yw.common.PlatformServiceException;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;
import java.util.zip.GZIPInputStream;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.compress.archivers.ArchiveException;
import org.apache.commons.compress.archivers.ArchiveStreamFactory;
import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.io.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import play.Environment;
import play.libs.Json;

@Slf4j
public class FileUtils {

  public static final Logger LOG = LoggerFactory.getLogger(FileUtils.class);

  public static String readResource(String filePath, Environment environment) {
    try (InputStream inputStream = environment.resourceAsStream(filePath)) {
      return IOUtils.toString(inputStream, StandardCharsets.UTF_8);
    } catch (IOException e) {
      throw new RuntimeException("Unable to read file " + filePath, e);
    }
  }

  public static void writeStringToFile(File file, String contents) throws Exception {
    try (FileWriter writer = new FileWriter(file)) {
      writer.write(contents);
    }
  }

  /**
   * Extracts the name and extension parts of a file name.
   *
   * <p>The resulting string is the rightmost characters of fullName, starting with the first
   * character after the path separator that separates the path information from the name and
   * extension.
   *
   * <p>The resulting string is equal to fullName, if fullName contains no path.
   *
   * @param fullName
   * @return
   */
  public static String getFileName(String fullName) {
    if (fullName == null) {
      return null;
    }
    int delimiterIndex = fullName.lastIndexOf(File.separatorChar);
    return delimiterIndex >= 0 ? fullName.substring(delimiterIndex + 1) : fullName;
  }

  public static String getFileChecksum(String file) throws IOException, NoSuchAlgorithmException {
    try (FileInputStream fis = new FileInputStream(file)) {
      byte[] byteArray = new byte[1024];
      int bytesCount = 0;

      MessageDigest digest = MessageDigest.getInstance("MD5");

      while ((bytesCount = fis.read(byteArray)) != -1) {
        digest.update(byteArray, 0, bytesCount);
      }

      fis.close();

      byte[] bytes = digest.digest();
      StringBuilder sb = new StringBuilder();
      for (byte b : bytes) {
        sb.append(Integer.toString((b & 0xff) + 0x100, 16).substring(1));
      }
      return sb.toString();
    }
  }

  public static List<File> listFiles(Path backupDir, String pattern) throws IOException {
    try (DirectoryStream<Path> directoryStream = Files.newDirectoryStream(backupDir, pattern)) {
      return StreamSupport.stream(directoryStream.spliterator(), false)
          .map(Path::toFile)
          .sorted(File::compareTo)
          .collect(Collectors.toList());
    } catch (NoSuchFileException e) {
      // Ignore error if the path does not exist.
    }
    return Collections.emptyList();
  }

  public static void moveFile(Path source, Path destination) throws IOException {
    Files.move(source, destination, REPLACE_EXISTING);
  }

  public static void writeJsonFile(String filePath, ArrayNode json) {
    writeFile(filePath, Json.prettyPrint(json));
  }

  public static void writeJsonFile(String filePath, JsonNode json) {
    writeFile(filePath, Json.prettyPrint(json));
  }

  public static void writeFile(String filePath, String contents) {
    try (FileWriter file = new FileWriter(filePath)) {
      file.write(contents);
      file.flush();
      LOG.info("Written: {}", filePath);
    } catch (IOException e) {
      LOG.error("Unable to write: {}", filePath);
      throw new RuntimeException(e.getMessage());
    }
  }

  /** deleteDirectory deletes entire directory recursively. */
  public static boolean deleteDirectory(File directoryToBeDeleted) {
    File[] allContents = directoryToBeDeleted.listFiles();
    if (allContents != null) {
      for (File file : allContents) {
        deleteDirectory(file);
      }
    }
    return directoryToBeDeleted.delete();
  }

  public static InputStream getInputStreamOrFail(File file) {
    return getInputStreamOrFail(file, false);
  }

  public static InputStream getInputStreamOrFail(File file, boolean deleteOnClose) {
    try {
      if (deleteOnClose) {
        return Files.newInputStream(file.toPath(), StandardOpenOption.DELETE_ON_CLOSE);
      }
      return new FileInputStream(file);
    } catch (IOException e) {
      throw new PlatformServiceException(INTERNAL_SERVER_ERROR, e.getMessage());
    }
  }

  /**
   * Returns the size of the file in bytes. Returns 0 in case the file doesn't exist.
   *
   * @param filePath path of the file
   * @return Size of file in bytes
   */
  public static long getFileSize(String filePath) {
    long fileSize;

    try {
      File file = new File(filePath);
      fileSize = file.length();
    } catch (Exception e) {
      LOG.error("Cannot open or get size of file with pathname " + filePath, e);
      fileSize = 0;
    }

    return fileSize;
  }

  public static Path getOrCreateTmpDirectory(String tmpDirectoryPath) {
    File tmpDir = new File(tmpDirectoryPath);
    if (!tmpDir.exists()) {
      LOG.info("Specified tmp directory {} does not exists creating now", tmpDirectoryPath);
      boolean created = tmpDir.mkdirs();
      if (created) {
        tmpDir.setExecutable(true, false);
        tmpDir.setWritable(true, false);
        tmpDir.setReadable(true, false);
      } else {
        throw new PlatformServiceException(
            INTERNAL_SERVER_ERROR,
            String.format("Failed to create tmp directory at path %s", tmpDirectoryPath));
      }
    } else {
      // Check if the provided path is directory & has read/write/execute permissions.
      if (!tmpDir.isDirectory()) {
        throw new PlatformServiceException(
            INTERNAL_SERVER_ERROR,
            String.format("Provided path %s is not a directory", tmpDirectoryPath));
      }

      if (!(tmpDir.canExecute() || tmpDir.canRead() || tmpDir.canWrite())) {
        throw new PlatformServiceException(
            INTERNAL_SERVER_ERROR,
            String.format(
                "Provided tmp directory %s does not have sufficient permissions",
                tmpDirectoryPath));
      }
    }

    return tmpDir.toPath();
  }

  public static List<String> listFiles(File file) throws IOException {
    return Files.walk(Paths.get(file.toURI()))
        .map(p -> p.toFile())
        .filter(f -> f.isFile())
        .map(f -> f.getAbsolutePath())
        .collect(Collectors.toList());
  }

  public static String computeHashForAFile(String content, int rSize)
      throws NoSuchAlgorithmException {
    MessageDigest md = MessageDigest.getInstance("SHA-256");
    byte[] hashBytes = md.digest(content.getBytes(StandardCharsets.UTF_8));

    StringBuilder hexString = new StringBuilder();
    for (byte b : hashBytes) {
      String hex = Integer.toHexString(0xff & b);
      if (hex.length() == 1) {
        hexString.append('0');
      }
      hexString.append(hex);
    }

    String hash = hexString.toString();
    if (hash.length() < rSize) {
      return hash;
    }
    return hexString.substring(0, rSize);
  }

  /**
   * Untar an input file into an output file.
   *
   * <p>The output file is created in the output folder, having the same name as the input file,
   * minus the '.tar' extension.
   *
   * @param inputFile the input .tar file
   * @param outputDir the output directory file.
   * @throws IOException
   * @throws FileNotFoundException
   * @return The {@link List} of {@link File}s with the untared content.
   * @throws ArchiveException
   */
  public static List<File> unTar(final File inputFile, final File outputDir)
      throws FileNotFoundException, IOException, ArchiveException {

    log.info(
        String.format(
            "Untaring %s to dir %s.", inputFile.getAbsolutePath(), outputDir.getAbsolutePath()));

    final List<File> untaredFiles = new LinkedList<File>();
    try (InputStream is = new FileInputStream(inputFile);
        TarArchiveInputStream debInputStream =
            (TarArchiveInputStream)
                new ArchiveStreamFactory().createArchiveInputStream("tar", is)) {
      TarArchiveEntry entry = null;
      while ((entry = (TarArchiveEntry) debInputStream.getNextEntry()) != null) {
        final File outputFile = new File(outputDir, entry.getName());
        if (entry.isDirectory()) {
          log.info(
              String.format(
                  "Attempting to write output directory %s.", outputFile.getAbsolutePath()));
          if (!outputFile.exists()) {
            log.info(
                String.format(
                    "Attempting to create output directory %s.", outputFile.getAbsolutePath()));
            if (!outputFile.mkdirs()) {
              throw new IllegalStateException(
                  String.format("Couldn't create directory %s.", outputFile.getAbsolutePath()));
            }
          }
        } else {
          // Don't log the output file here because platform logs get polluted.
          File parent = outputFile.getParentFile();
          if (!parent.exists()) parent.mkdirs();
          try (OutputStream outputFileStream = new FileOutputStream(outputFile)) {
            IOUtils.copy(debInputStream, outputFileStream);
          }
        }
        untaredFiles.add(outputFile);
      }
    }

    return untaredFiles;
  }

  /**
   * Ungzip an input file into an output file.
   *
   * <p>The output file is created in the output folder, having the same name as the input file,
   * minus the '.gz' extension.
   *
   * @param inputFile the input .gz file
   * @param outputDir the output directory file.
   * @throws IOException
   * @throws FileNotFoundException
   * @return The {@File} with the ungzipped content.
   */
  public static File unGzip(final File inputFile, final File outputDir)
      throws FileNotFoundException, IOException {

    log.info(
        String.format(
            "Ungzipping %s to dir %s.", inputFile.getAbsolutePath(), outputDir.getAbsolutePath()));

    final File outputFile =
        new File(outputDir, inputFile.getName().substring(0, inputFile.getName().length() - 3));

    try (GZIPInputStream in = new GZIPInputStream(new FileInputStream(inputFile));
        FileOutputStream out = new FileOutputStream(outputFile)) {
      IOUtils.copy(in, out);
      return outputFile;
    }
  }
}
