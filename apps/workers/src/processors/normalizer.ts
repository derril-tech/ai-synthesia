/**
 * Asset normalization and processing utilities
 */

import sharp from 'sharp';
import { execSync } from 'child_process';
import { createHash } from 'crypto';

export interface NormalizationOptions {
  stripMetadata?: boolean;
  resizeImages?: {
    maxWidth?: number;
    maxHeight?: number;
    quality?: number;
  };
  normalizeAudio?: {
    targetLoudness?: number; // LUFS
    sampleRate?: number;
    bitRate?: string;
  };
  textCleaning?: {
    removeExtraWhitespace?: boolean;
    normalizeLineEndings?: boolean;
    removeControlChars?: boolean;
  };
}

export class AssetNormalizer {
  /**
   * Normalize text content
   */
  static normalizeText(content: string, options: NormalizationOptions['textCleaning'] = {}): string {
    let normalized = content;

    if (options.removeControlChars !== false) {
      // Remove control characters except newlines and tabs
      normalized = normalized.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, '');
    }

    if (options.normalizeLineEndings !== false) {
      // Normalize line endings to LF
      normalized = normalized.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
    }

    if (options.removeExtraWhitespace !== false) {
      // Remove extra whitespace
      normalized = normalized
        .replace(/[ \t]+/g, ' ') // Multiple spaces/tabs to single space
        .replace(/\n\s*\n\s*\n/g, '\n\n') // Multiple newlines to double newline
        .trim();
    }

    return normalized;
  }

  /**
   * Normalize image content
   */
  static async normalizeImage(
    buffer: Buffer, 
    options: NormalizationOptions['resizeImages'] = {}
  ): Promise<Buffer> {
    let processor = sharp(buffer);

    // Strip EXIF and other metadata
    processor = processor.withMetadata({
      exif: {},
      icc: undefined,
      iptc: undefined,
      xmp: undefined,
    });

    // Resize if needed
    if (options.maxWidth || options.maxHeight) {
      processor = processor.resize({
        width: options.maxWidth,
        height: options.maxHeight,
        fit: 'inside',
        withoutEnlargement: true,
      });
    }

    // Set quality for JPEG
    const metadata = await sharp(buffer).metadata();
    if (metadata.format === 'jpeg') {
      processor = processor.jpeg({ 
        quality: options.quality || 85,
        progressive: true,
      });
    } else if (metadata.format === 'png') {
      processor = processor.png({ 
        quality: options.quality || 85,
        progressive: true,
      });
    } else if (metadata.format === 'webp') {
      processor = processor.webp({ 
        quality: options.quality || 85,
      });
    }

    return processor.toBuffer();
  }

  /**
   * Normalize audio content using FFmpeg
   */
  static async normalizeAudio(
    inputPath: string,
    outputPath: string,
    options: NormalizationOptions['normalizeAudio'] = {}
  ): Promise<void> {
    const targetLoudness = options.targetLoudness || -23; // EBU R128 standard
    const sampleRate = options.sampleRate || 44100;
    const bitRate = options.bitRate || '128k';

    try {
      // Build FFmpeg command
      const cmd = [
        'ffmpeg',
        '-i', `"${inputPath}"`,
        '-af', `loudnorm=I=${targetLoudness}:TP=-1.0:LRA=7.0`,
        '-ar', sampleRate.toString(),
        '-b:a', bitRate,
        '-y', // Overwrite output file
        `"${outputPath}"`
      ].join(' ');

      execSync(cmd, { stdio: 'pipe' });
    } catch (error) {
      console.error('Audio normalization failed:', error);
      throw new Error(`Failed to normalize audio: ${error}`);
    }
  }

  /**
   * Generate content hash for deduplication
   */
  static generateContentHash(buffer: Buffer): string {
    return createHash('sha256').update(buffer).digest('hex');
  }

  /**
   * Extract text from various document formats
   */
  static async extractText(buffer: Buffer, mimeType: string): Promise<string> {
    switch (mimeType) {
      case 'text/plain':
      case 'text/markdown':
        return buffer.toString('utf-8');
      
      case 'application/pdf':
        // Would use pdf-parse or similar library
        throw new Error('PDF text extraction not implemented');
      
      case 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        // Would use mammoth or similar library
        throw new Error('DOCX text extraction not implemented');
      
      default:
        throw new Error(`Unsupported document type: ${mimeType}`);
    }
  }

  /**
   * Validate and sanitize file content
   */
  static async validateContent(buffer: Buffer, expectedMimeType: string): Promise<{
    isValid: boolean;
    actualMimeType?: string;
    issues: string[];
  }> {
    const issues: string[] = [];
    
    // Check file size
    if (buffer.length === 0) {
      issues.push('File is empty');
      return { isValid: false, issues };
    }

    // Basic magic number checks
    const header = buffer.subarray(0, 16);
    let actualMimeType = expectedMimeType;

    // JPEG
    if (header[0] === 0xFF && header[1] === 0xD8) {
      actualMimeType = 'image/jpeg';
    }
    // PNG
    else if (header.subarray(0, 8).equals(Buffer.from([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]))) {
      actualMimeType = 'image/png';
    }
    // PDF
    else if (header.subarray(0, 4).toString() === '%PDF') {
      actualMimeType = 'application/pdf';
    }

    // Check if detected type matches expected
    if (actualMimeType !== expectedMimeType) {
      issues.push(`MIME type mismatch: expected ${expectedMimeType}, detected ${actualMimeType}`);
    }

    return {
      isValid: issues.length === 0,
      actualMimeType,
      issues,
    };
  }
}
