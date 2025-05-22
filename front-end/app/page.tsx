import Link from "next/link";

export default function Home() {
  return (
    <div className="container mx-auto p-8">
      <h1 className="text-4xl font-bold mb-6 text-center">
        Welcome to Model Mind Reader
      </h1>
      <p className="text-lg text-center mb-8">
        Explore the functionalities below to interact with the system.
      </p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Link
          href="/data-ingestion"
          className="block p-6 bg-blue-500 text-white rounded-lg shadow-md hover:bg-blue-600"
        >
          <h2 className="text-2xl font-bold mb-2">Data Ingestion</h2>
          <p>Upload and validate your datasets for analysis.</p>
        </Link>
        <Link
          href="/model-training"
          className="block p-6 bg-green-500 text-white rounded-lg shadow-md hover:bg-green-600"
        >
          <h2 className="text-2xl font-bold mb-2">Model Training</h2>
          <p>Train machine learning models with your data.</p>
        </Link>
        <Link
          href="/explainability"
          className="block p-6 bg-purple-500 text-white rounded-lg shadow-md hover:bg-purple-600"
        >
          <h2 className="text-2xl font-bold mb-2">Explainability</h2>
          <p>Generate explanations for model predictions.</p>
        </Link>
        <Link
          href="/fairness-metrics"
          className="block p-6 bg-red-500 text-white rounded-lg shadow-md hover:bg-red-600"
        >
          <h2 className="text-2xl font-bold mb-2">Fairness Metrics</h2>
          <p>Evaluate fairness metrics for your models.</p>
        </Link>
      </div>
    </div>
  );
}
