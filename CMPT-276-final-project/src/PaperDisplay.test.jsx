// src/PaperDisplay.test.jsx
import { test, beforeEach, afterEach, expect } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import PaperDisplay from './PaperDisplay';

// Use vi.fn instead of jest.fn
beforeEach(() => {
  global.fetch = vi.fn(() =>
    Promise.resolve({
      ok: true,
      json: () =>
        Promise.resolve({
          result: {
            verdict: "Verified",
            overall_confidence: 0.8,
            sub_claims: []
          }
        }),
    })
  );
});

afterEach(() => {
  vi.restoreAllMocks();
});

test('displays verification results when claim is provided', async () => {
  render(<PaperDisplay claim="Test Claim" />);
  
  // Wait until the verification result text appears
  await waitFor(() => {
    const verdictElement = screen.getByText(/Verified/i);
    expect(verdictElement).toBeInTheDocument();
  });
});
