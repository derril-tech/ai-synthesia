/**
 * Google Drive connector for asset ingestion
 */

import { google, drive_v3 } from 'googleapis';
import { GoogleAuth } from 'google-auth-library';
import { BaseConnector, ConnectorAsset, AssetMetadata } from './base';

export class GDriveConnector extends BaseConnector {
  private drive: drive_v3.Drive;
  private auth: GoogleAuth;

  constructor(config: any) {
    super(config);
    
    this.auth = new GoogleAuth({
      credentials: this.config.credentials,
      scopes: ['https://www.googleapis.com/auth/drive.readonly'],
    });
    
    this.drive = google.drive({ version: 'v3', auth: this.auth });
  }

  async testConnection(): Promise<boolean> {
    try {
      await this.drive.about.get({ fields: 'user' });
      return true;
    } catch (error) {
      console.error('GDrive connection test failed:', error);
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
    const pageSize = options?.limit || 100;
    const pageToken = options?.cursor;
    
    // Build query for supported file types
    const mimeTypes = [
      'text/plain',
      'text/markdown',
      'application/pdf',
      'image/jpeg',
      'image/png',
      'image/gif',
      'audio/mpeg',
      'audio/wav',
      'video/mp4',
      'video/quicktime',
    ];
    
    const query = `(${mimeTypes.map(type => `mimeType='${type}'`).join(' or ')}) and trashed=false`;

    try {
      const response = await this.drive.files.list({
        q: query,
        pageSize,
        pageToken,
        fields: 'nextPageToken, files(id, name, mimeType, size, createdTime, modifiedTime, description, webContentLink)',
      });

      const assets: ConnectorAsset[] = [];
      
      if (response.data.files) {
        for (const file of response.data.files) {
          if (!file.id || !file.name) continue;
          
          const asset: ConnectorAsset = {
            id: file.id,
            name: file.name,
            type: this.getAssetType(file.mimeType || ''),
            url: file.webContentLink || '',
            metadata: {
              filename: file.name,
              mimeType: file.mimeType || 'application/octet-stream',
              size: parseInt(file.size || '0'),
              source: 'gdrive',
              sourceId: file.id,
              description: file.description || undefined,
              createdAt: new Date(file.createdTime || Date.now()),
              modifiedAt: new Date(file.modifiedTime || Date.now()),
            },
          };
          
          assets.push(asset);
        }
      }

      return {
        assets,
        nextCursor: response.data.nextPageToken || undefined,
      };
    } catch (error) {
      console.error('Failed to list GDrive assets:', error);
      throw error;
    }
  }

  async downloadAsset(assetId: string): Promise<Buffer> {
    try {
      const response = await this.drive.files.get({
        fileId: assetId,
        alt: 'media',
      }, {
        responseType: 'arraybuffer',
      });

      return Buffer.from(response.data as ArrayBuffer);
    } catch (error) {
      console.error(`Failed to download GDrive asset ${assetId}:`, error);
      throw error;
    }
  }

  async getAssetMetadata(assetId: string): Promise<AssetMetadata> {
    try {
      const response = await this.drive.files.get({
        fileId: assetId,
        fields: 'id, name, mimeType, size, createdTime, modifiedTime, description',
      });

      const file = response.data;
      
      return {
        filename: file.name || 'unknown',
        mimeType: file.mimeType || 'application/octet-stream',
        size: parseInt(file.size || '0'),
        source: 'gdrive',
        sourceId: file.id || assetId,
        description: file.description || undefined,
        createdAt: new Date(file.createdTime || Date.now()),
        modifiedAt: new Date(file.modifiedTime || Date.now()),
      };
    } catch (error) {
      console.error(`Failed to get GDrive metadata for ${assetId}:`, error);
      throw error;
    }
  }

  private getAssetType(mimeType: string): 'text' | 'image' | 'audio' | 'video' | 'document' {
    if (mimeType.startsWith('text/')) return 'text';
    if (mimeType.startsWith('image/')) return 'image';
    if (mimeType.startsWith('audio/')) return 'audio';
    if (mimeType.startsWith('video/')) return 'video';
    return 'document';
  }
}
