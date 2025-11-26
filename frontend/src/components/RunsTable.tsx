// frontend/src/components/RunsTable.tsx
import React from 'react';
import Link from 'next/link';

interface Run {
  run_id: string;
  name: string;
  project: string;
  status: string;
  total_cost: number;
  avg_gpu_util: number;
  started_at: string;
}

interface RunsTableProps {
  runs: Run[];
  title: string;
}

const RunsTable: React.FC<RunsTableProps> = ({ runs, title }) => {
  if (runs.length === 0) {
    return <p className="text-gray-500">No training runs found.</p>;
  }

  return (
    <div className="bg-white shadow-xl rounded-lg overflow-hidden">
      <h2 className="text-xl font-semibold p-4 border-b text-gray-800">{title}</h2>
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Run Name</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Project</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Avg. GPU Util</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Cost</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Started At</th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {runs.map((run) => (
            <tr key={run.run_id} className="hover:bg-gray-50 transition duration-150">
              <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-indigo-600">
                <Link href={`/runs/${run.run_id}`} className="hover:underline">
                  {run.name}
                </Link>
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{run.project}</td>
              <td className="px-6 py-4 whitespace-nowrap">
                <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full 
                  ${run.status === 'completed' ? 'bg-green-100 text-green-800' : 
                    run.status === 'running' ? 'bg-yellow-100 text-yellow-800' : 'bg-red-100 text-red-800'}`}>
                  {run.status}
                </span>
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{run.avg_gpu_util ? `${run.avg_gpu_util.toFixed(1)}%` : 'N/A'}</td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{run.total_cost ? `$${run.total_cost.toFixed(2)}` : 'N/A'}</td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{new Date(run.started_at).toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default RunsTable;