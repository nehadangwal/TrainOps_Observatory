'use client';

import React from 'react';
import { Suspense } from 'react';

/**
 * Wrapper component to satisfy Next.js requirement for useSearchParams
 * to be wrapped in a Suspense boundary when used in Server Components.
 * This effectively prevents the server from trying to access client-only
 * APIs during the build process.
 */
export default function SuspenseWrapper({ children }: { children: React.ReactNode }) {
  // Use a fallback of null or a simple loading indicator if needed.
  return <Suspense fallback={null}>{children}</Suspense>;
}