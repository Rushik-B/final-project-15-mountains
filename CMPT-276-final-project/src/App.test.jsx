// src/App.test.jsx
import { test, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import App from './App';

test('renders Header and Body components', () => {
  render(<App />);
  // Check for an element that should be in the Header or Body
  const headerElement = screen.getByText(/Features/i);
  expect(headerElement).toBeInTheDocument();
});
