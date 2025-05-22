"use client";

import { useState } from "react";
import axios from "axios";

export default function DataIngestion() {
  const [fileContent, setFileContent] = useState<string | null>(null);
  const [targetColumn, setTargetColumn] = useState<string>("");
  const [modelType, setModelType] = useState<string>("Decision Tree");

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setFileContent(e.target?.result as string);
      };
      reader.readAsText(file);
    }
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();

    if (!fileContent) {
      alert("Please upload a CSV file.");
      return;
    }

    try {
      // Convert CSV to JSON
      const rows = fileContent.split("\n").map((row) => row.split(","));
      const headers = rows[0];
      const data = rows.slice(1).map((row) =>
        headers.reduce((acc, header, index) => {
          acc[header] = row[index];
          return acc;
        }, {} as Record<string, string>)
      );

      // Send JSON payload to the backend
      const response = await axios.post("http://127.0.0.1:8000/train_model", {
        data,
        target_column: targetColumn,
        model_type: modelType,
      });

      console.log("Response:", response.data);
      alert("Model trained successfully!");
    } catch (error) {
      console.error("Error:", error);
      alert("Failed to train the model.");
    }
  };

  return (
    <div className="container mx-auto p-8">
      <h1 className="text-3xl font-bold mb-6">Data Ingestion</h1>
      <form onSubmit={handleSubmit}>
        <div className="mb-4">
          <label className="block text-gray-700 font-bold mb-2">Upload CSV File:</label>
          <input
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            className="border rounded p-2 w-full"
          />
        </div>
        <div className="mb-4">
          <label className="block text-gray-700 font-bold mb-2">Target Column:</label>
          <input
            type="text"
            value={targetColumn}
            onChange={(e) => setTargetColumn(e.target.value)}
            className="border rounded p-2 w-full"
            placeholder="Enter the target column name"
          />
        </div>
        <div className="mb-4">
          <label className="block text-gray-700 font-bold mb-2">Model Type:</label>
          <select
            value={modelType}
            onChange={(e) => setModelType(e.target.value)}
            className="border rounded p-2 w-full"
          >
            <option value="Decision Tree">Decision Tree</option>
            <option value="Logistic Regression">Logistic Regression</option>
          </select>
        </div>
        <button
          type="submit"
          className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
        >
          Train Model
        </button>
      </form>
    </div>
  );
}