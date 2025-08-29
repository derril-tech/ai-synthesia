/**
 * YouTube connector for video/audio ingestion
 */

import youtubeDl from 'youtube-dl-exec';
import { BaseConnector, ConnectorAsset, AssetMetadata } from './base';

export class YouTubeConnector extends BaseConnector {
  async testConnection(): Promise<boolean> {
    try {
      // Test with a known public video
      await youtubeDl('https://www.youtube.com/watch?v=dQw4w9WgXcQ', {
        dumpSingleJson: true,
        noDownload: true,
      });
      return true;
    } catch (error) {
      console.error('YouTube connection test failed:', error);
      return false;
    }
  }

  async listAssets(options?: {
    limit?: number;
    cursor?: string;
    filter?: Record<string, any>;
  }): Promise<{
    assets: ConnectorAsset[];
    nextCursor?: string;
  }> {
    // YouTube listing requires specific URLs or channel/playlist IDs
    // This would typically be configured per workspace
    const urls = this.config.credentials.urls || [];
    const assets: ConnectorAsset[] = [];

    for (const url of urls) {
      try {
        const info = await youtubeDl(url, {
          dumpSingleJson: true,
          noDownload: true,
        });

        const asset: ConnectorAsset = {
          id: info.id,
          name: info.title || 'Unknown Video',
          type: 'video',
          url: info.webpage_url,
          metadata: {
            filename: `${info.title || 'video'}.${info.ext || 'mp4'}`,
            mimeType: this.getMimeType(info.ext || 'mp4'),
            size: info.filesize || 0,
            source: 'youtube',
            sourceId: info.id,
            description: info.description,
            tags: info.tags || [],
            createdAt: new Date(info.upload_date ? 
              `${info.upload_date.slice(0,4)}-${info.upload_date.slice(4,6)}-${info.upload_date.slice(6,8)}` : 
              Date.now()
            ),
            modifiedAt: new Date(info.upload_date ? 
              `${info.upload_date.slice(0,4)}-${info.upload_date.slice(4,6)}-${info.upload_date.slice(6,8)}` : 
              Date.now()
            ),
          },
        };

        assets.push(asset);
      } catch (error) {
        console.error(`Failed to get YouTube info for ${url}:`, error);
      }
    }

    return { assets };
  }

  async downloadAsset(assetId: string): Promise<Buffer> {
    try {
      // First get the video info to find the URL
      const urls = this.config.credentials.urls || [];
      let targetUrl = '';
      
      for (const url of urls) {
        const info = await youtubeDl(url, {
          dumpSingleJson: true,
          noDownload: true,
        });
        
        if (info.id === assetId) {
          targetUrl = url;
          break;
        }
      }

      if (!targetUrl) {
        throw new Error(`Asset ${assetId} not found in configured URLs`);
      }

      // Download the video/audio
      const output = await youtubeDl(targetUrl, {
        format: 'best[ext=mp4]/best',
        output: '-', // Output to stdout
      });

      return Buffer.from(output);
    } catch (error) {
      console.error(`Failed to download YouTube asset ${assetId}:`, error);
      throw error;
    }
  }

  async getAssetMetadata(assetId: string): Promise<AssetMetadata> {
    try {
      const urls = this.config.credentials.urls || [];
      
      for (const url of urls) {
        const info = await youtubeDl(url, {
          dumpSingleJson: true,
          noDownload: true,
        });
        
        if (info.id === assetId) {
          return {
            filename: `${info.title || 'video'}.${info.ext || 'mp4'}`,
            mimeType: this.getMimeType(info.ext || 'mp4'),
            size: info.filesize || 0,
            source: 'youtube',
            sourceId: info.id,
            description: info.description,
            tags: info.tags || [],
            createdAt: new Date(info.upload_date ? 
              `${info.upload_date.slice(0,4)}-${info.upload_date.slice(4,6)}-${info.upload_date.slice(6,8)}` : 
              Date.now()
            ),
            modifiedAt: new Date(info.upload_date ? 
              `${info.upload_date.slice(0,4)}-${info.upload_date.slice(4,6)}-${info.upload_date.slice(6,8)}` : 
              Date.now()
            ),
          };
        }
      }

      throw new Error(`Asset ${assetId} not found`);
    } catch (error) {
      console.error(`Failed to get YouTube metadata for ${assetId}:`, error);
      throw error;
    }
  }

  private getMimeType(ext: string): string {
    const mimeTypes: Record<string, string> = {
      'mp4': 'video/mp4',
      'webm': 'video/webm',
      'mkv': 'video/x-matroska',
      'mp3': 'audio/mpeg',
      'wav': 'audio/wav',
      'ogg': 'audio/ogg',
    };
    
    return mimeTypes[ext] || 'video/mp4';
  }
}
