/**
 * Base connector interface for external data sources
 */

export interface ConnectorConfig {
  credentials: Record<string, any>;
  workspaceId: string;
  userId: string;
}

export interface AssetMetadata {
  filename: string;
  mimeType: string;
  size: number;
  source: string;
  sourceId: string;
  tags?: string[];
  description?: string;
  createdAt: Date;
  modifiedAt: Date;
}

export interface ConnectorAsset {
  id: string;
  name: string;
  type: 'text' | 'image' | 'audio' | 'video' | 'document';
  url: string;
  metadata: AssetMetadata;
  content?: Buffer;
}

export abstract class BaseConnector {
  protected config: ConnectorConfig;

  constructor(config: ConnectorConfig) {
    this.config = config;
  }

  /**
   * Test connection to the external service
   */
  abstract testConnection(): Promise<boolean>;

  /**
   * List available assets from the source
   */
  abstract listAssets(options?: {
    limit?: number;
    cursor?: string;
    filter?: Record<string, any>;
  }): Promise<{
    assets: ConnectorAsset[];
    nextCursor?: string;
  }>;

  /**
   * Download asset content
   */
  abstract downloadAsset(assetId: string): Promise<Buffer>;

  /**
   * Get asset metadata without downloading content
   */
  abstract getAssetMetadata(assetId: string): Promise<AssetMetadata>;

  /**
   * Watch for changes (if supported)
   */
  watchChanges?(callback: (asset: ConnectorAsset) => void): Promise<void>;
}
