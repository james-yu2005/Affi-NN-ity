import { Client, type Client as GradioClient } from "@gradio/client";
import { NextResponse } from "next/server";

interface PredictRequest {
  smiles: string;
  fasta_full: string;
  fasta_pocket?: string;
}

function buildBaseUrl(): string {
  return process.env.HF_SPACE_URL?.replace(/\/$/, "")!;
}

function buildEndpoint(): string {
  return process.env.HF_PREDICT_ENDPOINT?.startsWith("/")
    ? process.env.HF_PREDICT_ENDPOINT
    : `/${process.env.HF_PREDICT_ENDPOINT}`;
}

let clientPromise: Promise<GradioClient> | null = null;

async function getGradioClient(): Promise<GradioClient> {
  if (!clientPromise) {
    clientPromise = Client.connect(buildBaseUrl());
  }
  return clientPromise;
}

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as PredictRequest;

    if (!body.smiles?.trim() || !body.fasta_full?.trim()) {
      return NextResponse.json(
        { error: "Both SMILES and full FASTA are required." },
        { status: 400 }
      );
    }

    const client = await getGradioClient();
    const endpoint = buildEndpoint();

    const result = await client.predict(endpoint, {
      smiles: body.smiles,
      fasta_full: body.fasta_full,
      fasta_pocket: body.fasta_pocket ?? "",
    });

    const predictedValue = Array.isArray(result.data)
      ? result.data[0]
      : (result.data as unknown);

    const predictedPkD =
      typeof predictedValue === "number"
        ? predictedValue
        : Number(predictedValue ?? NaN);

    if (Number.isNaN(predictedPkD)) {
      return NextResponse.json(
        { error: "Unexpected response from prediction endpoint." },
        { status: 500 }
      );
    }

    return NextResponse.json({
      predictedPkD,
      raw: result.data ?? null,
    });
  } catch (error) {
    console.error("HF predict error:", error);
    return NextResponse.json(
      {
        error:
          error instanceof Error ? error.message : "Prediction request failed.",
      },
      { status: 500 }
    );
  }
}
