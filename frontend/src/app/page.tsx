// frontend/src/app/compare/page.tsx
'use client';

import { useQuery } from '@tanstack/react-query';
import { useRouter, useSearchParams } from 'next/navigation';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000/api";

interface ComparisonRun {
  run_id: string;
  name: string;
  project: string;
  total_cost: number;
  avg_gpu_util: number;
  avg_throughput: number;
  bottleneck_type: string;
}

// Helper to fetch data for the comparison route
const fetchComparison = async (runIds: string[]) => {
  const params = new URLSearchParams();
  runIds.forEach(id => params.append('run_id', id));
  
  const response = await fetch(`${API_URL}/runs/compare?${params.toString()}`);
  if (!response.ok) throw new Error('Failed to fetch comparison data');
  const data = await response.json();
  return data.runs as ComparisonRun[];
};

const ComparePage: React.FC = () => {
  const searchParams = useSearchParams();
  const runIds = searchParams.getAll('run_id');

  const { data: comparisonData, isLoading, isError } = useQuery({
    queryKey: ['compareRuns', runIds],
    queryFn: () => fetchComparison(runIds),
    enabled: runIds.length === 2,
    refetchInterval: 30000, // Refetch every 30s
  });

  const run1 = comparisonData?.[0];
  const run2 = comparisonData?.[1];

  if (runIds.length !== 2) {
    return (
      <div className="p-10 text-center">
        <h1 className="text-2xl font-bold mb-4">Select Runs for Comparison</h1>
        <p className="text-gray-600">Please select exactly two runs from the dashboard to compare.</p>
      </div>
    );
  }

  if (isLoading) {
    return <div className="p-10 text-center text-lg">Loading Comparison Data...</div>;
  }
  
  if (isError || !run1 || !run2) {
    return <div className="p-10 text-center text-red-600">Error fetching comparison data. Check run IDs and API.</div>;
  }

  // --- Data Structuring for Bar Chart ---
  const barChartData = [
    { metric: 'Avg GPU Util (%)', [run1.name]: run1.avg_gpu_util, [run2.name]: run2.avg_gpu_util },
    { metric: 'Avg Throughput (samples/s)', [run1.name]: run1.avg_throughput, [run2.name]: run2.avg_throughput },
    { metric: 'Total Cost ($)', [run1.name]: run1.total_cost, [run2.name]: run2.total_cost },
  ];
  
  // Custom label for Bottleneck
  const bottleneckData = [
    { label: 'Run 1 Bottleneck', value: run1.bottleneck_type.toUpperCase().replace('_', ' '), run: run1.name },
    { label: 'Run 2 Bottleneck', value: run2.bottleneck_type.toUpperCase().replace('_', ' '), run: run2.name },
  ];
  
  return (
    <div className="max-w-7xl mx-auto py-10 px-4 sm:px-6 lg:px-8">
      <h1 className="text-3xl font-extrabold text-gray-900 mb-8">Run Comparison: {run1.name} vs {run2.name}</h1>
      
      {/* Side-by-Side Metadata */}
      <div className="grid grid-cols-2 gap-8 mb-12">
        <div className="p-6 bg-indigo-50 border-l-4 border-indigo-600 rounded-lg">
          <h2 className="text-xl font-semibold text-indigo-800">{run1.name}</h2>
          <p className="text-sm text-gray-600">ID: {run1.run_id.substring(0, 8)}...</p>
        </div>
        <div className="p-6 bg-purple-50 border-l-4 border-purple-600 rounded-lg">
          <h2 className="text-xl font-semibold text-purple-800">{run2.name}</h2>
          <p className="text-sm text-gray-600">ID: {run2.run_id.substring(0, 8)}...</p>
        </div>
      </div>

      {/* Comparison Chart */}
      <div className="mt-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-4">Performance Metrics </h2>
        <div className="bg-white p-6 rounded-lg shadow-xl h-[450px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={barChartData}
              margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="metric" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey={run1.name} fill="#4f46e5" name={run1.name} />
              <Bar dataKey={run2.name} fill="#9333ea" name={run2.name} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
      
      {/* Bottleneck Analysis Comparison */}
      <div className="mt-12">
        <h2 className="text-2xl font-bold text-gray-800 mb-4">Bottleneck Analysis</h2>
        <div className="grid grid-cols-2 gap-8">
          {bottleneckData.map((item, index) => (
            <div key={index} className="p-6 bg-gray-50 rounded-lg border border-gray-200">
              <p className="text-lg font-medium text-gray-700">{item.run}</p>
              <p className={`mt-2 text-2xl font-bold ${index === 0 ? 'text-indigo-600' : 'text-purple-600'}`}>
                {item.value}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ComparePage;