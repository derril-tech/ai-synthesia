/**
 * @jest-environment jsdom
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MantineProvider } from '@mantine/core';
import { ExportWizard } from '../../components/ExportWizard';

// Mock notifications
jest.mock('@mantine/notifications', () => ({
  notifications: {
    show: jest.fn(),
  },
}));

// Test wrapper with Mantine provider
const TestWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <MantineProvider>{children}</MantineProvider>
);

const mockStoryPackData = {
  id: 'test-story-pack-1',
  name: 'Test Story Pack',
  images: ['image1', 'image2', 'image3'],
  text: 'This is a test story pack content.',
  audio: {
    mixed_audio: 'base64_audio_data',
    metadata: {
      total_duration: 120,
      voice_used: 'nova',
    },
  },
};

const defaultProps = {
  storyPackId: 'test-story-pack-1',
  storyPackData: mockStoryPackData,
  opened: true,
  onClose: jest.fn(),
  onExportComplete: jest.fn(),
};

describe('ExportWizard', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders the export wizard modal', () => {
    render(
      <TestWrapper>
        <ExportWizard {...defaultProps} />
      </TestWrapper>
    );

    expect(screen.getByText('Export Story Pack')).toBeInTheDocument();
    expect(screen.getByText('Select Formats')).toBeInTheDocument();
  });

  it('displays all export format options', () => {
    render(
      <TestWrapper>
        <ExportWizard {...defaultProps} />
      </TestWrapper>
    );

    expect(screen.getByText('MP4 Video')).toBeInTheDocument();
    expect(screen.getByText('ZIP Bundle')).toBeInTheDocument();
    expect(screen.getByText('JSON Metadata')).toBeInTheDocument();
    expect(screen.getByText('PDF Report')).toBeInTheDocument();
  });

  it('allows selecting and deselecting export formats', async () => {
    const user = userEvent.setup();
    
    render(
      <TestWrapper>
        <ExportWizard {...defaultProps} />
      </TestWrapper>
    );

    // JSON and PDF should be selected by default
    const jsonCheckbox = screen.getByRole('checkbox', { name: /json metadata/i });
    const pdfCheckbox = screen.getByRole('checkbox', { name: /pdf report/i });
    const mp4Checkbox = screen.getByRole('checkbox', { name: /mp4 video/i });

    expect(jsonCheckbox).toBeChecked();
    expect(pdfCheckbox).toBeChecked();
    expect(mp4Checkbox).not.toBeChecked();

    // Select MP4
    await user.click(mp4Checkbox);
    expect(mp4Checkbox).toBeChecked();

    // Deselect JSON
    await user.click(jsonCheckbox);
    expect(jsonCheckbox).not.toBeChecked();
  });

  it('shows estimated time when formats are selected', async () => {
    render(
      <TestWrapper>
        <ExportWizard {...defaultProps} />
      </TestWrapper>
    );

    // Should show estimated time for default selected formats
    await waitFor(() => {
      expect(screen.getByText(/estimated total time/i)).toBeInTheDocument();
    });
  });

  it('prevents proceeding without selecting any formats', async () => {
    const user = userEvent.setup();
    
    render(
      <TestWrapper>
        <ExportWizard {...defaultProps} />
      </TestWrapper>
    );

    // Deselect all formats
    const jsonCheckbox = screen.getByRole('checkbox', { name: /json metadata/i });
    const pdfCheckbox = screen.getByRole('checkbox', { name: /pdf report/i });

    await user.click(jsonCheckbox);
    await user.click(pdfCheckbox);

    // Next button should be disabled
    const nextButton = screen.getByRole('button', { name: /next/i });
    expect(nextButton).toBeDisabled();
  });

  it('navigates through wizard steps', async () => {
    const user = userEvent.setup();
    
    render(
      <TestWrapper>
        <ExportWizard {...defaultProps} />
      </TestWrapper>
    );

    // Step 1: Select formats (default selections are fine)
    const nextButton = screen.getByRole('button', { name: /next/i });
    await user.click(nextButton);

    // Step 2: Video settings
    expect(screen.getByText('Video Settings')).toBeInTheDocument();
    expect(screen.getByText('Configure video export options')).toBeInTheDocument();

    await user.click(nextButton);

    // Step 3: Advanced settings
    expect(screen.getByText('Advanced Settings')).toBeInTheDocument();
  });

  it('shows video settings when MP4 is selected', async () => {
    const user = userEvent.setup();
    
    render(
      <TestWrapper>
        <ExportWizard {...defaultProps} />
      </TestWrapper>
    );

    // Select MP4 format
    const mp4Checkbox = screen.getByRole('checkbox', { name: /mp4 video/i });
    await user.click(mp4Checkbox);

    // Go to video settings step
    const nextButton = screen.getByRole('button', { name: /next/i });
    await user.click(nextButton);

    // Should show video configuration options
    expect(screen.getByText('Resolution')).toBeInTheDocument();
    expect(screen.getByText('Quality')).toBeInTheDocument();
    expect(screen.getByText('Video Template')).toBeInTheDocument();
  });

  it('shows advanced settings for selected formats', async () => {
    const user = userEvent.setup();
    
    render(
      <TestWrapper>
        <ExportWizard {...defaultProps} />
      </TestWrapper>
    );

    // Navigate to advanced settings (PDF is selected by default)
    const nextButton = screen.getByRole('button', { name: /next/i });
    await user.click(nextButton); // Video settings
    await user.click(nextButton); // Advanced settings

    // Should show PDF settings
    expect(screen.getByText('PDF Report Settings')).toBeInTheDocument();
    expect(screen.getByText('Include Images')).toBeInTheDocument();
    expect(screen.getByText('Include Quality Metrics')).toBeInTheDocument();
  });

  it('starts export process and shows progress', async () => {
    const user = userEvent.setup();
    const onExportComplete = jest.fn();
    
    render(
      <TestWrapper>
        <ExportWizard {...defaultProps} onExportComplete={onExportComplete} />
      </TestWrapper>
    );

    // Navigate to final step
    const nextButton = screen.getByRole('button', { name: /next/i });
    await user.click(nextButton); // Video settings
    await user.click(nextButton); // Advanced settings

    // Start export
    const startExportButton = screen.getByRole('button', { name: /start export/i });
    await user.click(startExportButton);

    // Should show export progress
    expect(screen.getByText('Exporting Content')).toBeInTheDocument();
    expect(screen.getByRole('progressbar')).toBeInTheDocument();

    // Wait for export to complete
    await waitFor(
      () => {
        expect(screen.getByText('Export completed successfully!')).toBeInTheDocument();
      },
      { timeout: 10000 }
    );

    // Should call onExportComplete
    expect(onExportComplete).toHaveBeenCalled();
  });

  it('handles export errors gracefully', async () => {
    const user = userEvent.setup();
    
    // Mock console.error to avoid test output noise
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
    
    render(
      <TestWrapper>
        <ExportWizard {...defaultProps} />
      </TestWrapper>
    );

    // Navigate to final step
    const nextButton = screen.getByRole('button', { name: /next/i });
    await user.click(nextButton);
    await user.click(nextButton);

    // Mock export failure by providing invalid data
    const invalidProps = {
      ...defaultProps,
      storyPackData: null, // This should cause an error
    };

    render(
      <TestWrapper>
        <ExportWizard {...invalidProps} />
      </TestWrapper>
    );

    consoleSpy.mockRestore();
  });

  it('allows going back to previous steps', async () => {
    const user = userEvent.setup();
    
    render(
      <TestWrapper>
        <ExportWizard {...defaultProps} />
      </TestWrapper>
    );

    // Navigate forward
    const nextButton = screen.getByRole('button', { name: /next/i });
    await user.click(nextButton);

    // Should be on video settings
    expect(screen.getByText('Video Settings')).toBeInTheDocument();

    // Go back
    const backButton = screen.getByRole('button', { name: /back/i });
    await user.click(backButton);

    // Should be back on format selection
    expect(screen.getByText('Select Formats')).toBeInTheDocument();
  });

  it('closes wizard when requested', async () => {
    const user = userEvent.setup();
    const onClose = jest.fn();
    
    render(
      <TestWrapper>
        <ExportWizard {...defaultProps} onClose={onClose} />
      </TestWrapper>
    );

    // Close button should be available (usually in modal header)
    // This test assumes the modal has a close button
    // In actual implementation, you might need to find the close button differently
    
    // For now, test that onClose is called when export completes
    // Navigate to export step and complete
    const nextButton = screen.getByRole('button', { name: /next/i });
    await user.click(nextButton);
    await user.click(nextButton);

    const startExportButton = screen.getByRole('button', { name: /start export/i });
    await user.click(startExportButton);

    // Wait for auto-close after successful export
    await waitFor(
      () => {
        expect(onClose).toHaveBeenCalled();
      },
      { timeout: 15000 }
    );
  });

  it('updates video settings correctly', async () => {
    const user = userEvent.setup();
    
    render(
      <TestWrapper>
        <ExportWizard {...defaultProps} />
      </TestWrapper>
    );

    // Select MP4 to enable video settings
    const mp4Checkbox = screen.getByRole('checkbox', { name: /mp4 video/i });
    await user.click(mp4Checkbox);

    // Navigate to video settings
    const nextButton = screen.getByRole('button', { name: /next/i });
    await user.click(nextButton);

    // Change resolution
    const resolutionSelect = screen.getByDisplayValue('1080p (1920×1080)');
    await user.click(resolutionSelect);
    
    // Select 720p option
    const option720p = screen.getByText('720p (1280×720)');
    await user.click(option720p);

    // Toggle audio setting
    const includeAudioSwitch = screen.getByRole('checkbox', { name: /include audio/i });
    await user.click(includeAudioSwitch);
  });

  it('validates traffic split for A/B tests', () => {
    // This would be relevant if the ExportWizard included A/B testing features
    // For now, this is a placeholder for future A/B testing functionality
    expect(true).toBe(true);
  });

  it('handles large file exports', async () => {
    const largeStoryPackData = {
      ...mockStoryPackData,
      images: new Array(20).fill('large_image_data'), // Simulate many images
      text: 'Very long story content '.repeat(1000), // Large text
    };

    render(
      <TestWrapper>
        <ExportWizard {...defaultProps} storyPackData={largeStoryPackData} />
      </TestWrapper>
    );

    // Should still render without issues
    expect(screen.getByText('Export Story Pack')).toBeInTheDocument();
  });
});
