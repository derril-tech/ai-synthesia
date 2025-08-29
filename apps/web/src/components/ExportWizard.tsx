'use client';

import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  Card,
  Checkbox,
  Group,
  Modal,
  Progress,
  Select,
  Stack,
  Text,
  Title,
  Badge,
  Alert,
  Stepper,
  NumberInput,
  Switch,
  Divider,
  ActionIcon,
  Tooltip,
} from '@mantine/core';
import { useForm } from '@mantine/form';
import { notifications } from '@mantine/notifications';
import { 
  IconDownload, 
  IconVideo, 
  IconFileZip, 
  IconFileText, 
  IconFile3d,
  IconSettings,
  IconCheck,
  IconX,
  IconInfoCircle
} from '@tabler/icons-react';

interface ExportOptions {
  formats: string[];
  videoSettings: {
    resolution: string;
    quality: string;
    template: string;
    includeAudio: boolean;
    includeCaptions: boolean;
  };
  pdfSettings: {
    includeImages: boolean;
    includeMetrics: boolean;
    includeRecommendations: boolean;
  };
  zipSettings: {
    includeRawAssets: boolean;
    includeReports: boolean;
    compressionLevel: number;
  };
}

interface ExportWizardProps {
  storyPackId: string;
  storyPackData: any;
  opened: boolean;
  onClose: () => void;
  onExportComplete: (exportData: any) => void;
}

const formatIcons = {
  mp4: IconVideo,
  zip: IconFileZip,
  json: IconFileText,
  pdf: IconFile3d,
};

const formatLabels = {
  mp4: 'MP4 Video',
  zip: 'ZIP Bundle',
  json: 'JSON Metadata',
  pdf: 'PDF Report',
};

const formatDescriptions = {
  mp4: 'Complete video with images, audio, and captions',
  zip: 'All assets and exports in a compressed bundle',
  json: 'Structured metadata and content information',
  pdf: 'Professional report with analysis and metrics',
};

export function ExportWizard({ 
  storyPackId, 
  storyPackData, 
  opened, 
  onClose, 
  onExportComplete 
}: ExportWizardProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [isExporting, setIsExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState(0);
  const [exportStatus, setExportStatus] = useState<string>('');
  const [estimatedTime, setEstimatedTime] = useState<Record<string, number>>({});

  const form = useForm<ExportOptions>({
    initialValues: {
      formats: ['json', 'pdf'],
      videoSettings: {
        resolution: '1920x1080',
        quality: 'high',
        template: 'story_pack',
        includeAudio: true,
        includeCaptions: true,
      },
      pdfSettings: {
        includeImages: true,
        includeMetrics: true,
        includeRecommendations: true,
      },
      zipSettings: {
        includeRawAssets: true,
        includeReports: true,
        compressionLevel: 6,
      },
    },
    validate: {
      formats: (value) => value.length === 0 ? 'Select at least one export format' : null,
    },
  });

  // Estimate export time when formats change
  useEffect(() => {
    if (form.values.formats.length > 0) {
      estimateExportTime();
    }
  }, [form.values.formats, form.values.videoSettings]);

  const estimateExportTime = async () => {
    try {
      // Simulate API call to estimate export time
      const estimates = {
        json: 1,
        pdf: 5,
        zip: 3,
        mp4: form.values.formats.includes('mp4') ? 
          (storyPackData?.images?.length || 1) * 8 + 15 : 0,
      };
      
      setEstimatedTime(estimates);
    } catch (error) {
      console.error('Failed to estimate export time:', error);
    }
  };

  const handleFormatToggle = (format: string, checked: boolean) => {
    const currentFormats = form.values.formats;
    if (checked) {
      form.setFieldValue('formats', [...currentFormats, format]);
    } else {
      form.setFieldValue('formats', currentFormats.filter(f => f !== format));
    }
  };

  const handleExport = async () => {
    if (!form.validate().hasErrors) {
      setIsExporting(true);
      setExportProgress(0);
      setCurrentStep(3); // Move to export step

      try {
        // Simulate export process with progress updates
        const totalSteps = form.values.formats.length;
        let completedSteps = 0;

        for (const format of form.values.formats) {
          setExportStatus(`Exporting ${formatLabels[format as keyof typeof formatLabels]}...`);
          
          // Simulate format-specific export time
          const formatTime = estimatedTime[format] || 5;
          const steps = Math.max(10, formatTime);
          
          for (let i = 0; i <= steps; i++) {
            await new Promise(resolve => setTimeout(resolve, formatTime * 100));
            const formatProgress = (completedSteps + (i / steps)) / totalSteps;
            setExportProgress(Math.round(formatProgress * 100));
          }
          
          completedSteps++;
        }

        setExportStatus('Finalizing export...');
        await new Promise(resolve => setTimeout(resolve, 1000));

        // Simulate successful export
        const exportResult = {
          exportId: `export_${Date.now()}`,
          formats: form.values.formats,
          files: form.values.formats.reduce((acc, format) => {
            acc[format] = `${storyPackId}_${format}_export`;
            return acc;
          }, {} as Record<string, string>),
          totalSize: '15.2 MB',
          exportTime: Object.values(estimatedTime).reduce((a, b) => a + b, 0),
        };

        setExportProgress(100);
        setExportStatus('Export completed successfully!');
        
        notifications.show({
          title: 'Export Complete',
          message: `Successfully exported ${form.values.formats.length} format(s)`,
          color: 'green',
          icon: <IconCheck size={16} />,
        });

        onExportComplete(exportResult);
        
        // Auto-close after success
        setTimeout(() => {
          onClose();
          resetWizard();
        }, 2000);

      } catch (error) {
        setExportStatus('Export failed');
        notifications.show({
          title: 'Export Failed',
          message: 'An error occurred during export. Please try again.',
          color: 'red',
          icon: <IconX size={16} />,
        });
      } finally {
        setIsExporting(false);
      }
    }
  };

  const resetWizard = () => {
    setCurrentStep(0);
    setIsExporting(false);
    setExportProgress(0);
    setExportStatus('');
    form.reset();
  };

  const nextStep = () => setCurrentStep((current) => Math.min(current + 1, 3));
  const prevStep = () => setCurrentStep((current) => Math.max(current - 1, 0));

  const getTotalEstimatedTime = () => {
    return form.values.formats.reduce((total, format) => {
      return total + (estimatedTime[format] || 0);
    }, 0);
  };

  const FormatSelectionStep = () => (
    <Stack gap="md">
      <Text size="sm" c="dimmed">
        Select the formats you want to export. Each format serves different purposes.
      </Text>
      
      <Stack gap="xs">
        {Object.entries(formatLabels).map(([format, label]) => {
          const Icon = formatIcons[format as keyof typeof formatIcons];
          const isSelected = form.values.formats.includes(format);
          
          return (
            <Card 
              key={format}
              withBorder
              style={{
                cursor: 'pointer',
                borderColor: isSelected ? 'var(--mantine-color-blue-5)' : undefined,
                backgroundColor: isSelected ? 'var(--mantine-color-blue-0)' : undefined,
              }}
              onClick={() => handleFormatToggle(format, !isSelected)}
            >
              <Group justify="space-between">
                <Group>
                  <Icon size={24} color={isSelected ? 'var(--mantine-color-blue-6)' : undefined} />
                  <Box>
                    <Text fw={500}>{label}</Text>
                    <Text size="sm" c="dimmed">
                      {formatDescriptions[format as keyof typeof formatDescriptions]}
                    </Text>
                  </Box>
                </Group>
                <Group>
                  {estimatedTime[format] && (
                    <Badge variant="light" size="sm">
                      ~{estimatedTime[format]}s
                    </Badge>
                  )}
                  <Checkbox
                    checked={isSelected}
                    onChange={(event) => handleFormatToggle(format, event.currentTarget.checked)}
                    onClick={(e) => e.stopPropagation()}
                  />
                </Group>
              </Group>
            </Card>
          );
        })}
      </Stack>

      {form.values.formats.length > 0 && (
        <Alert icon={<IconInfoCircle size={16} />} color="blue">
          <Text size="sm">
            Selected {form.values.formats.length} format(s). 
            Estimated total time: ~{getTotalEstimatedTime()} seconds
          </Text>
        </Alert>
      )}
    </Stack>
  );

  const VideoSettingsStep = () => (
    <Stack gap="md">
      <Text size="sm" c="dimmed">
        Configure video export settings (applies only if MP4 is selected).
      </Text>
      
      <Group grow>
        <Select
          label="Resolution"
          data={[
            { value: '1920x1080', label: '1080p (1920×1080)' },
            { value: '1280x720', label: '720p (1280×720)' },
            { value: '854x480', label: '480p (854×480)' },
          ]}
          {...form.getInputProps('videoSettings.resolution')}
        />
        
        <Select
          label="Quality"
          data={[
            { value: 'high', label: 'High (8 Mbps)' },
            { value: 'medium', label: 'Medium (5 Mbps)' },
            { value: 'low', label: 'Low (2 Mbps)' },
          ]}
          {...form.getInputProps('videoSettings.quality')}
        />
      </Group>

      <Select
        label="Video Template"
        description="Choose the style and pacing for your video"
        data={[
          { value: 'story_pack', label: 'Story Pack (Balanced pacing)' },
          { value: 'commercial', label: 'Commercial (Fast pacing)' },
          { value: 'educational', label: 'Educational (Slow pacing)' },
        ]}
        {...form.getInputProps('videoSettings.template')}
      />

      <Divider label="Audio & Captions" />

      <Group>
        <Switch
          label="Include Audio"
          description="Add narration and background music"
          {...form.getInputProps('videoSettings.includeAudio', { type: 'checkbox' })}
        />
        
        <Switch
          label="Include Captions"
          description="Add subtitle overlay to video"
          {...form.getInputProps('videoSettings.includeCaptions', { type: 'checkbox' })}
        />
      </Group>
    </Stack>
  );

  const AdvancedSettingsStep = () => (
    <Stack gap="md">
      <Text size="sm" c="dimmed">
        Configure advanced settings for PDF and ZIP exports.
      </Text>

      {form.values.formats.includes('pdf') && (
        <>
          <Title order={4}>PDF Report Settings</Title>
          <Stack gap="xs">
            <Switch
              label="Include Images"
              description="Add sample images to the PDF report"
              {...form.getInputProps('pdfSettings.includeImages', { type: 'checkbox' })}
            />
            <Switch
              label="Include Quality Metrics"
              description="Add evaluation scores and analysis"
              {...form.getInputProps('pdfSettings.includeMetrics', { type: 'checkbox' })}
            />
            <Switch
              label="Include Recommendations"
              description="Add improvement suggestions"
              {...form.getInputProps('pdfSettings.includeRecommendations', { type: 'checkbox' })}
            />
          </Stack>
        </>
      )}

      {form.values.formats.includes('zip') && (
        <>
          <Divider />
          <Title order={4}>ZIP Bundle Settings</Title>
          <Stack gap="xs">
            <Switch
              label="Include Raw Assets"
              description="Add original images, audio, and text files"
              {...form.getInputProps('zipSettings.includeRawAssets', { type: 'checkbox' })}
            />
            <Switch
              label="Include Reports"
              description="Add all generated reports and metadata"
              {...form.getInputProps('zipSettings.includeReports', { type: 'checkbox' })}
            />
            <NumberInput
              label="Compression Level"
              description="Higher values = smaller files, longer compression time"
              min={1}
              max={9}
              {...form.getInputProps('zipSettings.compressionLevel')}
            />
          </Stack>
        </>
      )}

      {!form.values.formats.includes('pdf') && !form.values.formats.includes('zip') && (
        <Alert color="gray">
          <Text size="sm">
            No advanced settings available for the selected formats.
          </Text>
        </Alert>
      )}
    </Stack>
  );

  const ExportProgressStep = () => (
    <Stack gap="md" align="center">
      <Title order={3}>Exporting Content</Title>
      
      <Box style={{ width: '100%', maxWidth: 400 }}>
        <Progress 
          value={exportProgress} 
          size="xl" 
          radius="md"
          striped={isExporting}
          animated={isExporting}
        />
        <Text ta="center" mt="xs" size="sm">
          {exportProgress}% Complete
        </Text>
      </Box>

      <Text ta="center" c="dimmed">
        {exportStatus}
      </Text>

      {exportProgress === 100 && (
        <Alert icon={<IconCheck size={16} />} color="green">
          Export completed successfully! The wizard will close automatically.
        </Alert>
      )}

      <Group>
        <Text size="sm" c="dimmed">
          Exporting: {form.values.formats.map(f => formatLabels[f as keyof typeof formatLabels]).join(', ')}
        </Text>
      </Group>
    </Stack>
  );

  return (
    <Modal
      opened={opened}
      onClose={onClose}
      title="Export Story Pack"
      size="lg"
      closeOnClickOutside={!isExporting}
      closeOnEscape={!isExporting}
    >
      <Stack gap="lg">
        <Stepper active={currentStep} onStepClick={setCurrentStep} allowNextStepsSelect={false}>
          <Stepper.Step 
            label="Select Formats" 
            description="Choose export formats"
            icon={<IconDownload size={18} />}
          >
            <FormatSelectionStep />
          </Stepper.Step>
          
          <Stepper.Step 
            label="Video Settings" 
            description="Configure video options"
            icon={<IconVideo size={18} />}
          >
            <VideoSettingsStep />
          </Stepper.Step>
          
          <Stepper.Step 
            label="Advanced Settings" 
            description="Fine-tune export options"
            icon={<IconSettings size={18} />}
          >
            <AdvancedSettingsStep />
          </Stepper.Step>
          
          <Stepper.Step 
            label="Export" 
            description="Generate files"
            icon={<IconCheck size={18} />}
          >
            <ExportProgressStep />
          </Stepper.Step>
        </Stepper>

        {currentStep < 3 && (
          <Group justify="space-between">
            <Button 
              variant="subtle" 
              onClick={prevStep}
              disabled={currentStep === 0}
            >
              Back
            </Button>
            
            <Group>
              {currentStep === 2 && (
                <Text size="sm" c="dimmed">
                  Ready to export {form.values.formats.length} format(s)
                </Text>
              )}
              
              {currentStep < 2 ? (
                <Button 
                  onClick={nextStep}
                  disabled={currentStep === 0 && form.values.formats.length === 0}
                >
                  Next
                </Button>
              ) : (
                <Button 
                  onClick={handleExport}
                  loading={isExporting}
                  disabled={form.values.formats.length === 0}
                >
                  Start Export
                </Button>
              )}
            </Group>
          </Group>
        )}

        {currentStep === 3 && exportProgress === 100 && (
          <Group justify="center">
            <Button onClick={() => { onClose(); resetWizard(); }}>
              Close
            </Button>
          </Group>
        )}
      </Stack>
    </Modal>
  );
}
