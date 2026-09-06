"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Upload, Database, Beaker, Zap, ArrowRight, FileText } from "lucide-react";
import DrugScreening from "@/components/DrugScreening";
import type { ScreeningResult } from "@/types/prediction";

export default function Home() {
  const [pdbContent, setPdbContent] = useState("");
  const [pdbFileName, setPdbFileName] = useState<string | null>(null);
  const [screeningResult, setScreeningResult] = useState<ScreeningResult | null>(null);

  const handlePdbFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".pdb")) {
      alert("Please upload a valid .pdb file");
      return;
    }

    const text = await file.text();
    setPdbContent(text);
    setPdbFileName(file.name);
  };

  const handleScreeningComplete = (result: ScreeningResult) => {
    setScreeningResult(result);
  };

  const hasPdb = pdbContent.trim().length > 0;

  return (
    <main className="min-h-screen relative overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 bg-gradient-to-br from-slate-900 via-blue-900 to-purple-900" />
      <div className="absolute inset-0 opacity-20" style={{
        backgroundImage: `radial-gradient(circle at 25% 25%, rgba(59, 130, 246, 0.1) 0%, transparent 50%),
                          radial-gradient(circle at 75% 75%, rgba(139, 92, 246, 0.1) 0%, transparent 50%)`
      }} />

      <div className="relative z-10 container mx-auto px-6 py-12">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <h1 className="text-6xl font-black bg-gradient-to-r from-slate-200 via-blue-200 to-slate-200 bg-clip-text text-transparent mb-4 tracking-tight">
            Affi-NN-ity
          </h1>
          <p className="text-xl font-medium text-slate-300 tracking-wide mb-4">
            High-Throughput Drug Screening Platform
          </p>
          <p className="text-slate-400 max-w-2xl mx-auto" />
        </motion.div>

        {/* Pipeline Visualization */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="mb-12"
        >
          <div className="bg-slate-800/40 backdrop-blur-sm border border-slate-600/40 rounded-2xl p-6">
            <div className="flex justify-center items-center gap-4 flex-wrap">
              {/* Step 1: Upload PDB */}
              <div className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${
                hasPdb ? "bg-emerald-500/20 border border-emerald-400/40" : "bg-slate-700/30 border border-slate-600/30"
              }`}>
                <Upload className={`w-5 h-5 ${hasPdb ? "text-emerald-400" : "text-slate-400"}`} />
                <div className="text-left">
                  <div className={`font-medium ${hasPdb ? "text-emerald-300" : "text-slate-300"}`}>
                    1. Upload PDB
                  </div>
                  <div className="text-xs text-slate-500">Protein structure</div>
                </div>
              </div>

              <ArrowRight className="w-4 h-4 text-slate-500" />

              {/* Step 2: Configure Screening */}
              <div className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${
                hasPdb ? "bg-blue-500/20 border border-blue-400/40" : "bg-slate-700/30 border border-slate-600/30"
              }`}>
                <Database className={`w-5 h-5 ${hasPdb ? "text-blue-400" : "text-slate-400"}`} />
                <div className="text-left">
                  <div className={`font-medium ${hasPdb ? "text-blue-300" : "text-slate-300"}`}>
                    2. Select Candidates
                  </div>
                  <div className="text-xs text-slate-500">50K / 100K / 200K</div>
                </div>
              </div>

              <ArrowRight className="w-4 h-4 text-slate-500" />

              {/* Step 3: Run Screening */}
              <div className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${
                screeningResult ? "bg-purple-500/20 border border-purple-400/40" : "bg-slate-700/30 border border-slate-600/30"
              }`}>
                <Beaker className={`w-5 h-5 ${screeningResult ? "text-purple-400" : "text-slate-400"}`} />
                <div className="text-left">
                  <div className={`font-medium ${screeningResult ? "text-purple-300" : "text-slate-300"}`}>
                    3. Screen Drugs
                  </div>
                  <div className="text-xs text-slate-500">Binding affinity</div>
                </div>
              </div>

              <ArrowRight className="w-4 h-4 text-slate-500" />

              {/* Step 4: Results */}
              <div className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${
                screeningResult ? "bg-emerald-500/20 border border-emerald-400/40" : "bg-slate-700/30 border border-slate-600/30"
              }`}>
                <Zap className={`w-5 h-5 ${screeningResult ? "text-emerald-400" : "text-slate-400"}`} />
                <div className="text-left">
                  <div className={`font-medium ${screeningResult ? "text-emerald-300" : "text-slate-300"}`}>
                    4. Top Candidates
                  </div>
                  <div className="text-xs text-slate-500">Ranked by pKd</div>
                </div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Main Content */}
        <div className="grid gap-8 lg:grid-cols-2">
          {/* Left Column - PDB Upload & Visualization */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="space-y-6"
          >
            {/* PDB Upload Card */}
            <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
              <div className="flex items-center gap-3 mb-6">
                <div className="p-3 bg-cyan-500/20 rounded-xl">
                  <FileText className="w-6 h-6 text-cyan-400" />
                </div>
                <div>
                  <h2 className="text-xl font-semibold text-white">Protein Structure</h2>
                  <p className="text-sm text-gray-400">Upload your PDB file to begin screening</p>
                </div>
              </div>

              {/* File Upload Area */}
              <div className={`border-2 border-dashed rounded-2xl p-8 text-center transition-all ${
                hasPdb 
                  ? "border-emerald-500/50 bg-emerald-500/5" 
                  : "border-gray-600/50 hover:border-gray-500/50"
              }`}>
                {hasPdb ? (
                  <div className="space-y-3">
                    <div className="w-16 h-16 bg-emerald-500/20 rounded-2xl flex items-center justify-center mx-auto">
                      <FileText className="w-8 h-8 text-emerald-400" />
                    </div>
                    <div>
                      <p className="text-emerald-300 font-semibold">{pdbFileName}</p>
                      <p className="text-sm text-gray-400">
                        {pdbContent.split("\n").filter(l => l.startsWith("ATOM") || l.startsWith("HETATM")).length} atoms loaded
                      </p>
                    </div>
                    <label
                      htmlFor="pdb-upload"
                      className="inline-block px-4 py-2 bg-gray-700/50 text-gray-300 rounded-xl cursor-pointer hover:bg-gray-600/50 transition text-sm"
                    >
                      Upload Different File
                    </label>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <Upload className="w-12 h-12 text-gray-500 mx-auto" />
                    <div>
                      <p className="text-gray-300 font-medium mb-1">Drop your PDB file here</p>
                      <p className="text-sm text-gray-500">or click to browse</p>
                    </div>
                    <label
                      htmlFor="pdb-upload"
                      className="inline-block px-6 py-3 bg-cyan-600/20 text-cyan-300 rounded-xl cursor-pointer hover:bg-cyan-600/30 transition font-medium border border-cyan-500/30"
                    >
                      Select PDB File
                    </label>
                  </div>
                )}
                <input
                  type="file"
                  accept=".pdb"
                  onChange={handlePdbFileUpload}
                  className="hidden"
                  id="pdb-upload"
                />
              </div>

              {/* Manual PDB Input */}
              <details className="mt-4">
                <summary className="text-sm text-gray-400 cursor-pointer hover:text-gray-300 transition">
                  Or paste PDB content manually
                </summary>
                <textarea
                  value={pdbContent}
                  onChange={(e) => {
                    setPdbContent(e.target.value);
                    setPdbFileName(null);
                  }}
                  placeholder="Paste PDB content here (ATOM/HETATM lines)..."
                  className="w-full h-32 mt-3 px-4 py-3 bg-black/30 border border-gray-700/50 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/30 font-mono text-xs"
                />
              </details>
            </div>

            {/* Dataset Info */}
            <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
              <h3 className="text-lg font-semibold text-white mb-4">ZINC Drug Database</h3>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div className="bg-black/20 rounded-xl p-3">
                  <p className="text-gray-400">Total Compounds</p>
                  <p className="text-2xl font-bold text-blue-300">~227K</p>
                </div>
                <div className="bg-black/20 rounded-xl p-3">
                  <p className="text-gray-400">Ranked By</p>
                  <p className="text-2xl font-bold text-emerald-300">QED</p>
                </div>
                <div className="bg-black/20 rounded-xl p-3">
                  <p className="text-gray-400">Data Fields</p>
                  <p className="text-lg font-semibold text-purple-300">SMILES, QED, MW, logP</p>
                </div>
                <div className="bg-black/20 rounded-xl p-3">
                  <p className="text-gray-400">Source</p>
                  <p className="text-lg font-semibold text-amber-300">Kaggle ZINC</p>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Right Column - Drug Screening */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
          >
            <DrugScreening
              pdbContent={pdbContent}
              onScreeningComplete={handleScreeningComplete}
            />
          </motion.div>
        </div>

        {/* Footer */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.6 }}
          className="mt-16 text-center"
        >
          <p className="text-sm text-slate-500">
            University of Waterloo | WAT.ai Research Initiative | Borealis AI Let&apos;s SOLVE It Program
          </p>
        </motion.div>
      </div>
    </main>
  );
}
