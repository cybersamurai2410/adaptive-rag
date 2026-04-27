import { render, screen } from '@testing-library/react';
import App from './App';

test('renders ask a question heading', () => {
  render(<App />);
  const headingElement = screen.getByText(/ask a question/i);
  expect(headingElement).toBeInTheDocument();
});
