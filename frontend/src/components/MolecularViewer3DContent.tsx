"use client";

import { useRef, useMemo, useState, useEffect, Suspense } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, PerspectiveCamera } from "@react-three/drei";
import * as THREE from "three";
import { motion, AnimatePresence } from "framer-motion";
import { Box, RotateCcw, Eye, EyeOff, Loader2 } from "lucide-react";
import type { MolecularViewer3DProps } from "./MolecularViewer3D.types";

interface Atom {
  x: number;
  y: number;
  z: number;
  element: string;
  chain: string;
  resSeq: number | null;
  resName: string;
}

interface Bond {
  start: number;
  end: number;
}

// Element colors based on CPK coloring convention
const ELEMENT_COLORS: Record<string, string> = {
  C: "#909090",
  N: "#3050F8",
  O: "#FF0D0D",
  S: "#FFFF30",
  P: "#FF8000",
  H: "#FFFFFF",
  F: "#90E050",
  Cl: "#1FF01F",
  Br: "#A62929",
  I: "#940094",
  Fe: "#E06633",
  Zn: "#7D80B0",
  Ca: "#3DFF00",
  Mg: "#8AFF00",
  Na: "#AB5CF2",
  K: "#8F40D4",
  default: "#FF69B4",
};

// Chain colors for distinguishing different chains
const CHAIN_COLORS = [
  "#22d3ee", // cyan
  "#c084fc", // purple
  "#f472b6", // pink
  "#34d399", // emerald
  "#f59e0b", // amber
  "#60a5fa", // blue
  "#a3e635", // lime
  "#fb7185", // rose
];

function parsePdb(pdb: string): { atoms: Atom[]; bonds: Bond[] } {
  const atoms: Atom[] = [];
  const bonds: Bond[] = [];
  
  const lines = pdb.split("\n");
  
  for (const line of lines) {
    if (line.startsWith("ATOM") || line.startsWith("HETATM")) {
      const x = parseFloat(line.slice(30, 38));
      const y = parseFloat(line.slice(38, 46));
      const z = parseFloat(line.slice(46, 54));
      const element = line.slice(76, 78).trim() || line.slice(12, 14).trim().replace(/\d/g, "") || "C";
      const chain = line.slice(21, 22).trim() || "A";
      const resSeqRaw = line.slice(22, 26).trim();
      const resSeq = resSeqRaw ? parseInt(resSeqRaw) : null;
      const resName = line.slice(17, 20).trim();
      
      if (!isNaN(x) && !isNaN(y) && !isNaN(z)) {
        atoms.push({ x, y, z, element, chain, resSeq, resName });
      }
    } else if (line.startsWith("CONECT")) {
      const parts = line.slice(6).trim().split(/\s+/).map(Number);
      const atomIndex = parts[0] - 1;
      for (let i = 1; i < parts.length; i++) {
        const bondedIndex = parts[i] - 1;
        if (bondedIndex > atomIndex && bondedIndex < atoms.length) {
          bonds.push({ start: atomIndex, end: bondedIndex });
        }
      }
    }
  }
  
  // Generate bonds based on distance if no CONECT records
  if (bonds.length === 0 && atoms.length > 1 && atoms.length < 5000) {
    const bondThreshold = 1.9; // Angstroms
    for (let i = 0; i < Math.min(atoms.length, 2000); i++) {
      for (let j = i + 1; j < Math.min(atoms.length, 2000); j++) {
        const dx = atoms[i].x - atoms[j].x;
        const dy = atoms[i].y - atoms[j].y;
        const dz = atoms[i].z - atoms[j].z;
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (dist < bondThreshold) {
          bonds.push({ start: i, end: j });
        }
      }
    }
  }
  
  return { atoms, bonds };
}

function getChainColor(chain: string): string {
  const index = chain.charCodeAt(0) % CHAIN_COLORS.length;
  return CHAIN_COLORS[index];
}

function AtomSphere({ 
  position, 
  element, 
  chain,
  colorMode 
}: { 
  position: [number, number, number]; 
  element: string;
  chain: string;
  colorMode: "element" | "chain";
}) {
  const color = colorMode === "element" 
    ? ELEMENT_COLORS[element] || ELEMENT_COLORS.default
    : getChainColor(chain);
    
  const radius = element === "H" ? 0.25 : 0.4;
  
  return (
    <mesh position={position}>
      <sphereGeometry args={[radius, 16, 16]} />
      <meshStandardMaterial 
        color={color} 
        metalness={0.3} 
        roughness={0.4}
      />
    </mesh>
  );
}

function BondCylinder({ 
  start, 
  end 
}: { 
  start: [number, number, number]; 
  end: [number, number, number];
}) {
  const midpoint = useMemo(() => [
    (start[0] + end[0]) / 2,
    (start[1] + end[1]) / 2,
    (start[2] + end[2]) / 2,
  ] as [number, number, number], [start, end]);
  
  const direction = useMemo(() => {
    const dir = new THREE.Vector3(
      end[0] - start[0],
      end[1] - start[1],
      end[2] - start[2]
    );
    return dir;
  }, [start, end]);
  
  const length = direction.length();
  
  const quaternion = useMemo(() => {
    const up = new THREE.Vector3(0, 1, 0);
    const q = new THREE.Quaternion();
    q.setFromUnitVectors(up, direction.clone().normalize());
    return q;
  }, [direction]);
  
  return (
    <mesh position={midpoint} quaternion={quaternion}>
      <cylinderGeometry args={[0.08, 0.08, length, 8]} />
      <meshStandardMaterial color="#666666" metalness={0.2} roughness={0.6} />
    </mesh>
  );
}

function MoleculeScene({ 
  atoms, 
  bonds,
  colorMode,
  autoRotate
}: { 
  atoms: Atom[]; 
  bonds: Bond[];
  colorMode: "element" | "chain";
  autoRotate: boolean;
}) {
  const groupRef = useRef<THREE.Group>(null);
  
  // Center the molecule
  const centeredAtoms = useMemo(() => {
    if (atoms.length === 0) return [];
    
    const sumX = atoms.reduce((acc, a) => acc + a.x, 0);
    const sumY = atoms.reduce((acc, a) => acc + a.y, 0);
    const sumZ = atoms.reduce((acc, a) => acc + a.z, 0);
    
    const center = {
      x: sumX / atoms.length,
      y: sumY / atoms.length,
      z: sumZ / atoms.length,
    };
    
    return atoms.map(a => ({
      ...a,
      x: a.x - center.x,
      y: a.y - center.y,
      z: a.z - center.z,
    }));
  }, [atoms]);
  
  useFrame((_, delta) => {
    if (groupRef.current && autoRotate) {
      groupRef.current.rotation.y += delta * 0.3;
    }
  });
  
  if (centeredAtoms.length === 0) return null;
  
  // Limit rendering for performance
  const maxAtoms = 5000;
  const displayAtoms = centeredAtoms.slice(0, maxAtoms);
  const displayBonds = bonds.filter(b => b.start < maxAtoms && b.end < maxAtoms);
  
  return (
    <group ref={groupRef}>
      {displayAtoms.map((atom, i) => (
        <AtomSphere 
          key={`atom-${i}`}
          position={[atom.x, atom.y, atom.z]}
          element={atom.element}
          chain={atom.chain}
          colorMode={colorMode}
        />
      ))}
      {displayBonds.map((bond, i) => {
        const startAtom = centeredAtoms[bond.start];
        const endAtom = centeredAtoms[bond.end];
        if (!startAtom || !endAtom) return null;
        return (
          <BondCylinder
            key={`bond-${i}`}
            start={[startAtom.x, startAtom.y, startAtom.z]}
            end={[endAtom.x, endAtom.y, endAtom.z]}
          />
        );
      })}
    </group>
  );
}

function CameraController({ atoms }: { atoms: Atom[] }) {
  const { camera } = useThree();
  
  useEffect(() => {
    if (atoms.length === 0) return;
    
    // Calculate bounding box
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;
    
    for (const atom of atoms) {
      minX = Math.min(minX, atom.x);
      maxX = Math.max(maxX, atom.x);
      minY = Math.min(minY, atom.y);
      maxY = Math.max(maxY, atom.y);
      minZ = Math.min(minZ, atom.z);
      maxZ = Math.max(maxZ, atom.z);
    }
    
    const sizeX = maxX - minX;
    const sizeY = maxY - minY;
    const sizeZ = maxZ - minZ;
    const maxSize = Math.max(sizeX, sizeY, sizeZ);
    
    // Position camera based on molecule size
    const distance = Math.max(maxSize * 1.5, 30);
    camera.position.set(distance * 0.7, distance * 0.5, distance);
    camera.lookAt(0, 0, 0);
  }, [atoms, camera]);
  
  return null;
}

function LoadingSpinner() {
  return (
    <div className="absolute inset-0 flex items-center justify-center bg-black/50">
      <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
    </div>
  );
}

export default function MolecularViewer3DContent({ 
  pdbContent, 
  isVisible, 
  onToggle 
}: MolecularViewer3DProps) {
  const [colorMode, setColorMode] = useState<"element" | "chain">("element");
  const [autoRotate, setAutoRotate] = useState(true);
  
  const { atoms, bonds } = useMemo(() => {
    if (!pdbContent.trim()) return { atoms: [], bonds: [] };
    return parsePdb(pdbContent);
  }, [pdbContent]);
  
  const hasContent = atoms.length > 0;
  
  // Stats
  const chains = useMemo(() => {
    const uniqueChains = new Set(atoms.map(a => a.chain));
    return Array.from(uniqueChains);
  }, [atoms]);
  
  return (
    <div className="space-y-3">
      {/* Toggle Button */}
      <button
        onClick={onToggle}
        disabled={!pdbContent.trim()}
        className={`w-full flex items-center justify-center gap-2 px-4 py-3 rounded-xl font-semibold transition-all ${
          !pdbContent.trim()
            ? "bg-gray-700/30 text-gray-500 cursor-not-allowed"
            : isVisible
            ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
            : "bg-gray-700/50 text-gray-300 hover:bg-gray-700/70 border border-gray-600/50"
        }`}
      >
        {isVisible ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
        <span>{isVisible ? "Hide 3D Viewer" : "Show 3D Viewer"}</span>
        {hasContent && (
          <span className="text-xs opacity-70">
            ({atoms.length} atoms, {chains.length} chain{chains.length !== 1 ? "s" : ""})
          </span>
        )}
      </button>
      
      {/* 3D Viewer */}
      <AnimatePresence>
        {isVisible && hasContent && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden"
          >
            <div className="bg-black/40 rounded-2xl border border-cyan-500/30 overflow-hidden">
              {/* Controls Bar */}
              <div className="flex items-center justify-between px-4 py-2 bg-black/30 border-b border-cyan-500/20">
                <div className="flex items-center gap-2 text-xs text-gray-400">
                  <Box className="w-4 h-4 text-cyan-400" />
                  <span>3D Molecular Structure</span>
                </div>
                
                <div className="flex items-center gap-2">
                  {/* Color Mode Toggle */}
                  <button
                    onClick={() => setColorMode(m => m === "element" ? "chain" : "element")}
                    className="px-3 py-1 rounded-lg text-xs bg-gray-700/50 text-gray-300 hover:bg-gray-600/50 transition"
                  >
                    Color: {colorMode === "element" ? "Element" : "Chain"}
                  </button>
                  
                  {/* Auto Rotate Toggle */}
                  <button
                    onClick={() => setAutoRotate(!autoRotate)}
                    className={`p-1.5 rounded-lg transition ${
                      autoRotate 
                        ? "bg-cyan-500/20 text-cyan-300" 
                        : "bg-gray-700/50 text-gray-400"
                    }`}
                    title={autoRotate ? "Stop rotation" : "Start rotation"}
                  >
                    <RotateCcw className="w-4 h-4" />
                  </button>
                </div>
              </div>
              
              {/* Canvas */}
              <div className="h-[400px] relative">
                <Suspense fallback={<LoadingSpinner />}>
                  <Canvas>
                    <PerspectiveCamera makeDefault position={[50, 30, 50]} fov={50} />
                    <CameraController atoms={atoms} />
                    <OrbitControls 
                      enablePan={true}
                      enableZoom={true}
                      enableRotate={true}
                      autoRotate={false}
                    />
                    
                    {/* Lighting */}
                    <ambientLight intensity={0.4} />
                    <directionalLight position={[10, 10, 5]} intensity={0.8} />
                    <directionalLight position={[-10, -10, -5]} intensity={0.3} />
                    <pointLight position={[0, 20, 0]} intensity={0.5} />
                    
                    {/* Molecule */}
                    <MoleculeScene 
                      atoms={atoms} 
                      bonds={bonds}
                      colorMode={colorMode}
                      autoRotate={autoRotate}
                    />
                  </Canvas>
                </Suspense>
              </div>
              
              {/* Stats Bar */}
              <div className="px-4 py-2 bg-black/30 border-t border-cyan-500/20 flex items-center justify-between text-xs text-gray-400">
                <div className="flex items-center gap-4">
                  <span>Atoms: <span className="text-cyan-300">{atoms.length}</span></span>
                  <span>Bonds: <span className="text-cyan-300">{bonds.length}</span></span>
                  <span>Chains: <span className="text-cyan-300">{chains.join(", ") || "A"}</span></span>
                </div>
                <div className="text-gray-500">
                  Drag to rotate • Scroll to zoom
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      
      {/* No Content Message */}
      {isVisible && !hasContent && pdbContent.trim() && (
        <div className="p-4 bg-amber-500/10 border border-amber-500/30 rounded-xl text-amber-200 text-sm">
          Could not parse any atoms from the PDB content. Make sure it contains valid ATOM/HETATM records.
        </div>
      )}
    </div>
  );
}
