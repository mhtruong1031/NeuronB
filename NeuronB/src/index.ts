import { z } from "zod";
import axios from "axios";
import { defineDAINService, ToolConfig } from "@dainprotocol/service-sdk";
import {
  LayoutUIBuilder,
  ImageCardUIBuilder,
  CardUIBuilder,
  DainResponse
} from "@dainprotocol/utils";

const port = Number(process.env.PORT) || 2022;
const analysisServerUrl = process.env.ANALYSIS_SERVER_URL || "http://localhost:8000/analyze";

interface AnalysisResult {
  images: {
    front: string;
    side: string;
    bottom: string;
    isometric: string;
  };
}

const tumorVisualizationTool: ToolConfig = {
  id: "tumor-visualization-analysis",
  name: "Tumor Visualization",
  description: "Upload brain MRI and get segmented 3D tumor views.",
  input: z
    .object({
      flairUrl: z.string().url().describe("URL to FLAIR MRI .nii.gz file"),
      t1gdUrl: z.string().url().describe("URL to T1Gd MRI .nii.gz file"),
      prompt: z.string().describe("Optional prompt for DAIN to interpret the result."),
    })
    .describe("Brain MRI scan URLs and analysis prompt"),
  output: z
    .object({
      images: z.object({
        front: z.string(),
        side: z.string(),
        bottom: z.string(),
        isometric: z.string(),
      }),
    })
    .describe("Rendered tumor visualization views and make sure to include an in depth analysis of possible implications, which parts of the brain are affected, and future steps."),

  handler: async ({ flairUrl, t1gdUrl, prompt }, agentInfo) => {
    try {
      console.log(`Processing request for agent ${agentInfo.id}`);
      const [flairResponse, t1gdResponse] = await Promise.all([
        axios.get(flairUrl, { responseType: 'arraybuffer' }),
        axios.get(t1gdUrl, { responseType: 'arraybuffer' })
      ]);

      const formData = new FormData();
      formData.append("flair", new Blob([flairResponse.data]), "flair.nii.gz");
      formData.append("t1gd", new Blob([t1gdResponse.data]), "t1gd.nii.gz");
      formData.append("prompt", prompt);
      
      console.log("Posting to:", analysisServerUrl);
      const res = await axios.post<AnalysisResult>(analysisServerUrl, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      const { images } = res.data;

      return new DainResponse({
        text: `🧠 Tumor Visualization Complete.`,
        data: { images },
        ui: new LayoutUIBuilder()
          .setRenderMode("page")
          .setLayoutType("grid")
          .setColumns(2)
          .addChild(new ImageCardUIBuilder(images.front).title("Front View").build())
          .addChild(new ImageCardUIBuilder(images.side).title("Side View").build())
          .addChild(new ImageCardUIBuilder(images.bottom).title("Bottom View").build())
          .addChild(new ImageCardUIBuilder(images.isometric).title("Isometric View").build())
          .addChild(new ImageCardUIBuilder("https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png").build())
          .build(),
      });
    } catch (error) {
      console.error("Error in tumor visualization:", error);
      return new DainResponse({
        text: "An error occurred during tumor visualization.",
        data: { error: error instanceof Error ? error.message : String(error) },
        ui: new CardUIBuilder()
          .title("Error")
          .content("Failed to process MRI files. Please check the URLs and try again.")
          .build(),
      });
    }
  },
};

const dainService = defineDAINService({
  metadata: {
    title: "NeuronB",
    description: "Generate a 3D mesh of brain tumor MRI imaging",
    version: "1.0.0",
    author: "Minh Truong",
    tags: ["neuro", "brain", "dain", "mri"],
    logo: "https://cdn-icons-png.flaticon.com/512/252/252035.png",
  },
  identity: {
    apiKey: process.env.DAIN_API_KEY,
  },
  tools: [tumorVisualizationTool],
});

dainService.startNode({ port }).then(({ address }) => {
  console.log(`NeuronB is running at http://localhost:${address().port}`);
});
