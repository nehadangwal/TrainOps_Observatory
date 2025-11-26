'use client';

import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// Create a client instance outside of the component to avoid re-creation on render
const queryClient = new QueryClient();

/**
 * Provides the TanStack Query client to the entire application.
 * This should wrap the content in the RootLayout.
 */
export default function QueryProvider({ children }: { children: React.ReactNode }) {
  return (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  );
}