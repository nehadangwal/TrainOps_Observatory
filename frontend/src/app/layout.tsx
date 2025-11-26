import "./globals.css"; // Assuming globals.css is in the app directory
import QueryProvider from "../components/QueryProvider"; 
import SuspenseWrapper from "./SuspenseWrapper"; 

export const metadata = {
  title: 'TrainOps Observatory',
  description: 'AI Training Run Monitoring and Analysis',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-gray-50 min-h-screen">
        {/*
          Wrap the entire application content with SuspenseWrapper.
          This handles the useSearchParams issue reported during the build.
        */}
        <SuspenseWrapper>
          <QueryProvider>
            <div className="container mx-auto p-4 md:p-8">
              {children}
            </div>
          </QueryProvider>
        </SuspenseWrapper>
      </body>
    </html>
  );
}