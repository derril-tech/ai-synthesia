'use client';

import React, { useState } from 'react';
import {
  Box,
  Button,
  Card,
  ColorInput,
  Grid,
  Group,
  Select,
  Stack,
  Text,
  TextInput,
  Textarea,
  Title,
  Badge,
  ActionIcon,
  Modal,
} from '@mantine/core';
import { useForm } from '@mantine/form';
import { notifications } from '@mantine/notifications';
import { IconPlus, IconTrash, IconEye, IconPalette } from '@tabler/icons-react';

interface ColorPalette {
  primary: string;
  secondary: string;
  accent: string;
  background: string;
  text: string;
  muted: string;
}

interface Typography {
  heading_font: string;
  body_font: string;
  mono_font: string;
  heading_sizes: Record<string, string>;
  line_heights: Record<string, number>;
}

interface Lexicon {
  preferred_terms: string[];
  avoid_terms: string[];
  tone_keywords: string[];
  brand_voice: string;
}

interface BrandKitFormData {
  name: string;
  color_palette: ColorPalette;
  typography: Typography;
  tone_guidelines: string;
  lexicon: Lexicon;
  logo_url: string;
}

interface BrandKitEditorProps {
  initialData?: Partial<BrandKitFormData>;
  onSave: (data: BrandKitFormData) => Promise<void>;
  onCancel: () => void;
}

const defaultColorPalette: ColorPalette = {
  primary: '#3B82F6',
  secondary: '#6B7280',
  accent: '#F59E0B',
  background: '#FFFFFF',
  text: '#1F2937',
  muted: '#9CA3AF',
};

const defaultTypography: Typography = {
  heading_font: 'Inter',
  body_font: 'Inter',
  mono_font: 'JetBrains Mono',
  heading_sizes: {
    h1: '2.5rem',
    h2: '2rem',
    h3: '1.5rem',
    h4: '1.25rem',
  },
  line_heights: {
    heading: 1.2,
    body: 1.6,
    tight: 1.4,
  },
};

const defaultLexicon: Lexicon = {
  preferred_terms: [],
  avoid_terms: [],
  tone_keywords: [],
  brand_voice: 'professional',
};

export function BrandKitEditor({ initialData, onSave, onCancel }: BrandKitEditorProps) {
  const [previewOpen, setPreviewOpen] = useState(false);
  const [loading, setLoading] = useState(false);

  const form = useForm<BrandKitFormData>({
    initialValues: {
      name: initialData?.name || '',
      color_palette: initialData?.color_palette || defaultColorPalette,
      typography: initialData?.typography || defaultTypography,
      tone_guidelines: initialData?.tone_guidelines || '',
      lexicon: initialData?.lexicon || defaultLexicon,
      logo_url: initialData?.logo_url || '',
    },
    validate: {
      name: (value) => (value.length < 1 ? 'Name is required' : null),
    },
  });

  const handleSubmit = async (values: BrandKitFormData) => {
    setLoading(true);
    try {
      await onSave(values);
      notifications.show({
        title: 'Success',
        message: 'Brand kit saved successfully',
        color: 'green',
      });
    } catch (error) {
      notifications.show({
        title: 'Error',
        message: 'Failed to save brand kit',
        color: 'red',
      });
    } finally {
      setLoading(false);
    }
  };

  const addTerm = (field: 'preferred_terms' | 'avoid_terms' | 'tone_keywords', term: string) => {
    if (term.trim()) {
      const currentTerms = form.values.lexicon[field];
      if (!currentTerms.includes(term.trim())) {
        form.setFieldValue(`lexicon.${field}`, [...currentTerms, term.trim()]);
      }
    }
  };

  const removeTerm = (field: 'preferred_terms' | 'avoid_terms' | 'tone_keywords', index: number) => {
    const currentTerms = form.values.lexicon[field];
    form.setFieldValue(`lexicon.${field}`, currentTerms.filter((_, i) => i !== index));
  };

  const PreviewModal = () => (
    <Modal
      opened={previewOpen}
      onClose={() => setPreviewOpen(false)}
      title="Brand Kit Preview"
      size="lg"
    >
      <Stack gap="md">
        {/* Color Palette Preview */}
        <Box>
          <Title order={4} mb="sm">Color Palette</Title>
          <Grid>
            {Object.entries(form.values.color_palette).map(([name, color]) => (
              <Grid.Col span={4} key={name}>
                <Box
                  style={{
                    backgroundColor: color,
                    height: 60,
                    borderRadius: 8,
                    border: '1px solid #e0e0e0',
                  }}
                />
                <Text size="sm" mt={4} tt="capitalize">{name}</Text>
                <Text size="xs" c="dimmed">{color}</Text>
              </Grid.Col>
            ))}
          </Grid>
        </Box>

        {/* Typography Preview */}
        <Box>
          <Title order={4} mb="sm">Typography</Title>
          <Stack gap="xs">
            <Text
              style={{
                fontFamily: form.values.typography.heading_font,
                fontSize: form.values.typography.heading_sizes.h1,
                lineHeight: form.values.typography.line_heights.heading,
                color: form.values.color_palette.text,
              }}
            >
              Heading Example
            </Text>
            <Text
              style={{
                fontFamily: form.values.typography.body_font,
                lineHeight: form.values.typography.line_heights.body,
                color: form.values.color_palette.text,
              }}
            >
              This is body text using the selected typography settings. It demonstrates how your content will look with the chosen fonts and spacing.
            </Text>
          </Stack>
        </Box>

        {/* Brand Voice */}
        <Box>
          <Title order={4} mb="sm">Brand Voice</Title>
          <Badge variant="light" color={form.values.color_palette.primary}>
            {form.values.lexicon.brand_voice}
          </Badge>
        </Box>
      </Stack>
    </Modal>
  );

  return (
    <Box>
      <form onSubmit={form.onSubmit(handleSubmit)}>
        <Stack gap="xl">
          {/* Header */}
          <Group justify="space-between">
            <Title order={2}>Brand Kit Editor</Title>
            <Group>
              <Button
                variant="light"
                leftSection={<IconEye size={16} />}
                onClick={() => setPreviewOpen(true)}
              >
                Preview
              </Button>
              <Button variant="subtle" onClick={onCancel}>
                Cancel
              </Button>
              <Button type="submit" loading={loading}>
                Save Brand Kit
              </Button>
            </Group>
          </Group>

          {/* Basic Information */}
          <Card withBorder>
            <Stack gap="md">
              <Title order={3}>Basic Information</Title>
              <TextInput
                label="Brand Kit Name"
                placeholder="Enter a name for this brand kit"
                {...form.getInputProps('name')}
              />
              <TextInput
                label="Logo URL"
                placeholder="https://example.com/logo.png"
                {...form.getInputProps('logo_url')}
              />
            </Stack>
          </Card>

          {/* Color Palette */}
          <Card withBorder>
            <Stack gap="md">
              <Group>
                <IconPalette size={20} />
                <Title order={3}>Color Palette</Title>
              </Group>
              <Grid>
                {Object.entries(form.values.color_palette).map(([colorName, colorValue]) => (
                  <Grid.Col span={6} key={colorName}>
                    <ColorInput
                      label={colorName.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      {...form.getInputProps(`color_palette.${colorName}`)}
                    />
                  </Grid.Col>
                ))}
              </Grid>
            </Stack>
          </Card>

          {/* Typography */}
          <Card withBorder>
            <Stack gap="md">
              <Title order={3}>Typography</Title>
              <Grid>
                <Grid.Col span={4}>
                  <Select
                    label="Heading Font"
                    data={['Inter', 'Roboto', 'Open Sans', 'Lato', 'Montserrat']}
                    {...form.getInputProps('typography.heading_font')}
                  />
                </Grid.Col>
                <Grid.Col span={4}>
                  <Select
                    label="Body Font"
                    data={['Inter', 'Roboto', 'Open Sans', 'Lato', 'Source Sans Pro']}
                    {...form.getInputProps('typography.body_font')}
                  />
                </Grid.Col>
                <Grid.Col span={4}>
                  <Select
                    label="Monospace Font"
                    data={['JetBrains Mono', 'Fira Code', 'Source Code Pro', 'Monaco']}
                    {...form.getInputProps('typography.mono_font')}
                  />
                </Grid.Col>
              </Grid>
            </Stack>
          </Card>

          {/* Brand Voice & Lexicon */}
          <Card withBorder>
            <Stack gap="md">
              <Title order={3}>Brand Voice & Lexicon</Title>
              
              <Select
                label="Brand Voice"
                data={[
                  'professional',
                  'friendly',
                  'authoritative',
                  'casual',
                  'playful',
                  'sophisticated',
                  'approachable',
                ]}
                {...form.getInputProps('lexicon.brand_voice')}
              />

              <Textarea
                label="Tone Guidelines"
                placeholder="Describe the tone and style guidelines for your brand..."
                minRows={3}
                {...form.getInputProps('tone_guidelines')}
              />

              {/* Preferred Terms */}
              <Box>
                <Text size="sm" fw={500} mb="xs">Preferred Terms</Text>
                <Group mb="xs">
                  {form.values.lexicon.preferred_terms.map((term, index) => (
                    <Badge
                      key={index}
                      variant="light"
                      rightSection={
                        <ActionIcon
                          size="xs"
                          color="red"
                          radius="xl"
                          variant="transparent"
                          onClick={() => removeTerm('preferred_terms', index)}
                        >
                          <IconTrash size={10} />
                        </ActionIcon>
                      }
                    >
                      {term}
                    </Badge>
                  ))}
                </Group>
                <TextInput
                  placeholder="Add preferred term..."
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault();
                      addTerm('preferred_terms', e.currentTarget.value);
                      e.currentTarget.value = '';
                    }
                  }}
                />
              </Box>

              {/* Terms to Avoid */}
              <Box>
                <Text size="sm" fw={500} mb="xs">Terms to Avoid</Text>
                <Group mb="xs">
                  {form.values.lexicon.avoid_terms.map((term, index) => (
                    <Badge
                      key={index}
                      variant="light"
                      color="red"
                      rightSection={
                        <ActionIcon
                          size="xs"
                          color="red"
                          radius="xl"
                          variant="transparent"
                          onClick={() => removeTerm('avoid_terms', index)}
                        >
                          <IconTrash size={10} />
                        </ActionIcon>
                      }
                    >
                      {term}
                    </Badge>
                  ))}
                </Group>
                <TextInput
                  placeholder="Add term to avoid..."
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault();
                      addTerm('avoid_terms', e.currentTarget.value);
                      e.currentTarget.value = '';
                    }
                  }}
                />
              </Box>
            </Stack>
          </Card>
        </Stack>
      </form>

      <PreviewModal />
    </Box>
  );
}
