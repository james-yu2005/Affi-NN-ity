"use client";

import { useState, useEffect } from "react";
import dynamic from "next/dynamic";
import { Loader2 } from "lucide-react";
import type { MolecularViewer3DProps } from "./MolecularViewer3D.types";

// Loading component for the 3D viewer
function LoadingPlaceholder() {
  return (
    <div className="w-full flex items-center justify-center gap-2 px-4 py-3 rounded-xl bg-gray-700/30 text-gray-400">
      <Loader2 className="w-4 h-4 animate-spin" />
      <span>Loading 3D Viewer...</span>
    </div>
  );
}

// Dynamically import the 3D viewer with SSR disabled
// This prevents the ReactCurrentOwner error from @react-three/fiber
const MolecularViewer3DContent = dynamic(
  () => import("./MolecularViewer3DContent"),
  {
    ssr: false,
    loading: LoadingPlaceholder,
  }
);

export default function MolecularViewer3D(props: MolecularViewer3DProps) {
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
  }, []);

  // Don't render the dynamic component until we're on the client
  if (!isMounted) {
    return <LoadingPlaceholder />;
  }

  return <MolecularViewer3DContent {...props} />;
}

export type { MolecularViewer3DProps };
