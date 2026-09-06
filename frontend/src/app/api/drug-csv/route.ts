import { NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

interface DrugRow {
  smiles: string;
  qed: number;
  mw: number | null;
  logP: number | null;
}

let cachedDrugs: DrugRow[] | null = null;

function parseCsv(content: string): string[][] {
  const rows: string[][] = [];
  let current = "";
  const row: string[] = [];
  let inQuotes = false;

  for (let i = 0; i < content.length; i++) {
    const char = content[i];
    const next = content[i + 1];

    if (char === '"') {
      if (inQuotes && next === '"') {
        current += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if ((char === "\n" || char === "\r") && !inQuotes) {
      if (char === "\r" && next === "\n") {
        i++;
      }
      row.push(current);
      if (row.length) rows.push([...row]);
      row.length = 0;
      current = "";
    } else if (char === "," && !inQuotes) {
      row.push(current);
      current = "";
    } else {
      current += char;
    }
  }

  // push final field
  row.push(current);
  if (row.length > 1 || row[0]) {
    rows.push([...row]);
  }

  return rows;
}

async function loadDrugsFromCSV(): Promise<DrugRow[]> {
  if (cachedDrugs) {
    return cachedDrugs;
  }

  const sortedPath = path.join(process.cwd(), "kaggle_zinc_filtered_sorted.csv");
  const basePath = path.join(process.cwd(), "kaggle_zinc_filtered.csv");
  const csvPath = await fs
    .access(sortedPath)
    .then(() => sortedPath)
    .catch(() => basePath);
  const content = await fs.readFile(csvPath, "utf-8");

  const rows = parseCsv(content);
  if (!rows.length) {
    throw new Error("CSV is empty or could not be parsed.");
  }

  const header = rows[0].map((h) => h.trim());
  const smilesIdx = header.findIndex((h) => h.toLowerCase() === "smiles");
  const qedIdx = header.findIndex((h) => h.toLowerCase() === "qed");
  const mwIdx = header.findIndex((h) => h.toLowerCase() === "mw");
  const logPIdx = header.findIndex((h) => h.toLowerCase() === "logp");

  if (smilesIdx === -1 || qedIdx === -1) {
    throw new Error("CSV missing required columns: smiles, qed");
  }

  const drugs: DrugRow[] = [];

  for (let i = 1; i < rows.length; i++) {
    const row = rows[i];
    if (!row.length) continue;

    const smiles = row[smilesIdx]?.replace(/[\n\r]/g, "").trim();
    const qed = parseFloat(row[qedIdx] ?? "");
    const mw = mwIdx >= 0 ? parseFloat(row[mwIdx] ?? "") : null;
    const logP = logPIdx >= 0 ? parseFloat(row[logPIdx] ?? "") : null;

    if (smiles && !isNaN(qed)) {
      drugs.push({
        smiles,
        qed,
        mw: mw !== null && !isNaN(mw) ? mw : null,
        logP: logP !== null && !isNaN(logP) ? logP : null,
      });
    }
  }

  // Already sorted CSV, but ensure descending order
  drugs.sort((a, b) => b.qed - a.qed);

  cachedDrugs = drugs;
  return drugs;
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const limitParam = searchParams.get("limit");
  const limit = limitParam ? Math.min(Number(limitParam), 227000) : 50000;

  try {
    const allDrugs = await loadDrugsFromCSV();
    const selectedDrugs = allDrugs.slice(0, limit);

    return NextResponse.json({
      count: selectedDrugs.length,
      totalAvailable: allDrugs.length,
      drugs: selectedDrugs,
    });
  } catch (error) {
    console.error("Drug CSV loading error:", error);
    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : "Failed to load drug candidates from CSV.",
      },
      { status: 500 }
    );
  }
}
