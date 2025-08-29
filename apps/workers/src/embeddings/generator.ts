/**
 * Multimodal embedding generation for semantic search and similarity
 */

import OpenAI from 'openai';
import axios from 'axios';

export interface EmbeddingResult {
  embedding: number[];
  model: string;
  dimensions: number;
}

export class EmbeddingGenerator {
  private openai: OpenAI;

  constructor(apiKey: string) {
    this.openai = new OpenAI({ apiKey });
  }

  /**
   * Generate text embeddings using OpenAI's ada-002
   */
  async generateTextEmbedding(text: string): Promise<EmbeddingResult> {
    try {
      const response = await this.openai.embeddings.create({
        model: 'text-embedding-ada-002',
        input: text,
      });

      const embedding = response.data[0].embedding;
      
      return {
        embedding,
        model: 'text-embedding-ada-002',
        dimensions: embedding.length,
      };
    } catch (error) {
      console.error('Failed to generate text embedding:', error);
      throw error;
    }
  }

  /**
   * Generate image embeddings using CLIP (via Hugging Face API)
   */
  async generateImageEmbedding(imageBuffer: Buffer): Promise<EmbeddingResult> {
    try {
      // Convert image to base64
      const base64Image = imageBuffer.toString('base64');
      
      // Use Hugging Face CLIP model
      const response = await axios.post(
        'https://api-inference.huggingface.co/models/openai/clip-vit-base-patch32',
        {
          inputs: base64Image,
        },
        {
          headers: {
            'Authorization': `Bearer ${process.env.HUGGINGFACE_API_KEY}`,
            'Content-Type': 'application/json',
          },
        }
      );

      const embedding = response.data;
      
      return {
        embedding,
        model: 'clip-vit-base-patch32',
        dimensions: embedding.length,
      };
    } catch (error) {
      console.error('Failed to generate image embedding:', error);
      throw error;
    }
  }

  /**
   * Generate audio embeddings using CLAP (Contrastive Language-Audio Pre-training)
   */
  async generateAudioEmbedding(audioBuffer: Buffer): Promise<EmbeddingResult> {
    try {
      // For now, we'll use a text description of the audio
      // In production, you'd use a proper CLAP model
      const audioDescription = await this.extractAudioFeatures(audioBuffer);
      return this.generateTextEmbedding(audioDescription);
    } catch (error) {
      console.error('Failed to generate audio embedding:', error);
      throw error;
    }
  }

  /**
   * Generate multimodal embeddings for content with multiple modalities
   */
  async generateMultimodalEmbedding(content: {
    text?: string;
    image?: Buffer;
    audio?: Buffer;
  }): Promise<EmbeddingResult> {
    const embeddings: number[][] = [];
    let model = 'multimodal';

    // Generate embeddings for each modality
    if (content.text) {
      const textEmb = await this.generateTextEmbedding(content.text);
      embeddings.push(textEmb.embedding);
    }

    if (content.image) {
      const imageEmb = await this.generateImageEmbedding(content.image);
      embeddings.push(imageEmb.embedding);
    }

    if (content.audio) {
      const audioEmb = await this.generateAudioEmbedding(content.audio);
      embeddings.push(audioEmb.embedding);
    }

    if (embeddings.length === 0) {
      throw new Error('No content provided for embedding generation');
    }

    // Combine embeddings using weighted average
    const combinedEmbedding = this.combineEmbeddings(embeddings);

    return {
      embedding: combinedEmbedding,
      model,
      dimensions: combinedEmbedding.length,
    };
  }

  /**
   * Calculate similarity between two embeddings
   */
  calculateSimilarity(embedding1: number[], embedding2: number[]): number {
    if (embedding1.length !== embedding2.length) {
      throw new Error('Embeddings must have the same dimensions');
    }

    // Cosine similarity
    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;

    for (let i = 0; i < embedding1.length; i++) {
      dotProduct += embedding1[i] * embedding2[i];
      norm1 += embedding1[i] * embedding1[i];
      norm2 += embedding2[i] * embedding2[i];
    }

    return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
  }

  /**
   * Find similar content using vector search
   */
  async findSimilarContent(
    queryEmbedding: number[],
    candidateEmbeddings: { id: string; embedding: number[] }[],
    threshold: number = 0.7,
    limit: number = 10
  ): Promise<{ id: string; similarity: number }[]> {
    const similarities = candidateEmbeddings.map(candidate => ({
      id: candidate.id,
      similarity: this.calculateSimilarity(queryEmbedding, candidate.embedding),
    }));

    return similarities
      .filter(item => item.similarity >= threshold)
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, limit);
  }

  /**
   * Combine multiple embeddings into a single embedding
   */
  private combineEmbeddings(embeddings: number[][]): number[] {
    if (embeddings.length === 0) {
      throw new Error('No embeddings to combine');
    }

    const dimensions = embeddings[0].length;
    const combined = new Array(dimensions).fill(0);

    // Weighted average (equal weights for now)
    for (const embedding of embeddings) {
      for (let i = 0; i < dimensions; i++) {
        combined[i] += embedding[i] / embeddings.length;
      }
    }

    return combined;
  }

  /**
   * Extract audio features for embedding generation
   * This is a placeholder - in production you'd use proper audio analysis
   */
  private async extractAudioFeatures(audioBuffer: Buffer): Promise<string> {
    // Placeholder: return generic audio description
    // In production, you'd analyze tempo, pitch, genre, etc.
    return 'audio content with musical elements and speech';
  }

  /**
   * Batch process embeddings for multiple items
   */
  async batchGenerateEmbeddings(
    items: Array<{
      id: string;
      content: {
        text?: string;
        image?: Buffer;
        audio?: Buffer;
      };
    }>
  ): Promise<Array<{
    id: string;
    embedding: EmbeddingResult;
  }>> {
    const results = [];

    // Process in batches to avoid rate limits
    const batchSize = 5;
    for (let i = 0; i < items.length; i += batchSize) {
      const batch = items.slice(i, i + batchSize);
      
      const batchPromises = batch.map(async item => {
        try {
          const embedding = await this.generateMultimodalEmbedding(item.content);
          return { id: item.id, embedding };
        } catch (error) {
          console.error(`Failed to generate embedding for item ${item.id}:`, error);
          return null;
        }
      });

      const batchResults = await Promise.all(batchPromises);
      results.push(...batchResults.filter(result => result !== null));

      // Small delay between batches
      if (i + batchSize < items.length) {
        await new Promise(resolve => setTimeout(resolve, 100));
      }
    }

    return results as Array<{ id: string; embedding: EmbeddingResult }>;
  }
}
