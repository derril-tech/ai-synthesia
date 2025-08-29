/**
 * Notion connector for content ingestion
 */

import { Client } from '@notionhq/client';
import { BaseConnector, ConnectorAsset, AssetMetadata } from './base';

export class NotionConnector extends BaseConnector {
  private notion: Client;

  constructor(config: any) {
    super(config);
    
    this.notion = new Client({
      auth: this.config.credentials.token,
    });
  }

  async testConnection(): Promise<boolean> {
    try {
      await this.notion.users.me();
      return true;
    } catch (error) {
      console.error('Notion connection test failed:', error);
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
    const startCursor = options?.cursor;

    try {
      const response = await this.notion.search({
        filter: {
          property: 'object',
          value: 'page',
        },
        page_size: pageSize,
        start_cursor: startCursor,
      });

      const assets: ConnectorAsset[] = [];

      for (const page of response.results) {
        if (page.object !== 'page') continue;
        
        // Get page content
        const blocks = await this.notion.blocks.children.list({
          block_id: page.id,
        });

        // Extract text content from blocks
        const textContent = this.extractTextFromBlocks(blocks.results);
        
        const asset: ConnectorAsset = {
          id: page.id,
          name: this.getPageTitle(page) || 'Untitled',
          type: 'text',
          url: (page as any).url || '',
          metadata: {
            filename: `${this.getPageTitle(page) || 'untitled'}.md`,
            mimeType: 'text/markdown',
            size: Buffer.byteLength(textContent, 'utf8'),
            source: 'notion',
            sourceId: page.id,
            createdAt: new Date((page as any).created_time),
            modifiedAt: new Date((page as any).last_edited_time),
          },
          content: Buffer.from(textContent, 'utf8'),
        };

        assets.push(asset);
      }

      return {
        assets,
        nextCursor: response.next_cursor || undefined,
      };
    } catch (error) {
      console.error('Failed to list Notion assets:', error);
      throw error;
    }
  }

  async downloadAsset(assetId: string): Promise<Buffer> {
    try {
      // Get page blocks
      const blocks = await this.notion.blocks.children.list({
        block_id: assetId,
      });

      const textContent = this.extractTextFromBlocks(blocks.results);
      return Buffer.from(textContent, 'utf8');
    } catch (error) {
      console.error(`Failed to download Notion asset ${assetId}:`, error);
      throw error;
    }
  }

  async getAssetMetadata(assetId: string): Promise<AssetMetadata> {
    try {
      const page = await this.notion.pages.retrieve({ page_id: assetId });
      
      return {
        filename: `${this.getPageTitle(page) || 'untitled'}.md`,
        mimeType: 'text/markdown',
        size: 0, // Will be calculated when content is downloaded
        source: 'notion',
        sourceId: page.id,
        createdAt: new Date((page as any).created_time),
        modifiedAt: new Date((page as any).last_edited_time),
      };
    } catch (error) {
      console.error(`Failed to get Notion metadata for ${assetId}:`, error);
      throw error;
    }
  }

  private getPageTitle(page: any): string | null {
    if (page.properties?.title?.title?.[0]?.plain_text) {
      return page.properties.title.title[0].plain_text;
    }
    if (page.properties?.Name?.title?.[0]?.plain_text) {
      return page.properties.Name.title[0].plain_text;
    }
    return null;
  }

  private extractTextFromBlocks(blocks: any[]): string {
    let content = '';
    
    for (const block of blocks) {
      switch (block.type) {
        case 'paragraph':
          content += this.extractRichText(block.paragraph.rich_text) + '\n\n';
          break;
        case 'heading_1':
          content += '# ' + this.extractRichText(block.heading_1.rich_text) + '\n\n';
          break;
        case 'heading_2':
          content += '## ' + this.extractRichText(block.heading_2.rich_text) + '\n\n';
          break;
        case 'heading_3':
          content += '### ' + this.extractRichText(block.heading_3.rich_text) + '\n\n';
          break;
        case 'bulleted_list_item':
          content += '- ' + this.extractRichText(block.bulleted_list_item.rich_text) + '\n';
          break;
        case 'numbered_list_item':
          content += '1. ' + this.extractRichText(block.numbered_list_item.rich_text) + '\n';
          break;
        case 'code':
          content += '```\n' + this.extractRichText(block.code.rich_text) + '\n```\n\n';
          break;
        case 'quote':
          content += '> ' + this.extractRichText(block.quote.rich_text) + '\n\n';
          break;
      }
    }
    
    return content.trim();
  }

  private extractRichText(richText: any[]): string {
    return richText.map(text => text.plain_text).join('');
  }
}
