export default function Explainability() {
  return (
    <div className="container mx-auto p-8">
      <h1 className="text-2xl font-bold mb-4">Explainability</h1>
      <textarea
        placeholder="Enter input data for explanation"
        className="w-full p-2 border rounded mb-4"
      />
      <button
        className="bg-purple-500 text-white px-4 py-2 rounded hover:bg-purple-600"
      >
        Generate Explanation
      </button>
    </div>
  );
}
