import path from "node:path";
import fs from "node:fs";
import { spawn } from "node:child_process";
import { Readable } from "node:stream";
import type { AlignmentClip } from "./alignment";

export const repositoryRoot = path.resolve(process.cwd(), "..");
const outputRoot = process.env.ALIGNMENT_DATA_ROOT
  ? path.resolve(process.env.ALIGNMENT_DATA_ROOT)
  : path.join(repositoryRoot, "data/review");
const sources = [
  { id: "granite-control", title: "Granite", description: "35.1 seconds · first half", video: "source-full.webm" },
  { id: "butte-h2-slot2", title: "Butte United", description: "30 seconds · second half", video: "source.webm" },
];
export function clipRoot(id: string): string {
  if (!sources.some(s => s.id === id)) throw new Error("Unknown review clip.");
  return path.join(outputRoot, id);
}
export function alignmentClips(): AlignmentClip[] {
  return sources.map(({id,title,description,video}) => ({id,title,description,ready: fs.existsSync(path.join(clipRoot(id), "packet.json")) && fs.existsSync(path.join(clipRoot(id),video))}));
}

export async function runAlignment(command: "state" | "save" | "locate", id: string, body?: unknown): Promise<unknown> {
  const root = clipRoot(id);
  return new Promise((resolve, reject) => {
    const child = spawn(path.join(repositoryRoot, ".venv/bin/python"), [path.join(repositoryRoot,"calibration/tools/field_adjustment.py"), command, "--root", root], { cwd: repositoryRoot, env: {...process.env, OPENBLAS_NUM_THREADS: "1"}, stdio: ["pipe","pipe","pipe"] });
    let output = "", error = "";
    const timeout = setTimeout(() => { child.kill("SIGTERM"); reject(new Error("The local tool did not finish responding. Reload the saved alignment before trying again.")); }, 240000);
    child.stdout.on("data", (b: Buffer) => { output += b.toString(); if(output.length > 8_000_000) child.kill("SIGTERM"); });
    child.stderr.on("data", (b: Buffer) => { error = (error + b.toString()).slice(-2000); });
    child.on("error", () => { clearTimeout(timeout); reject(new Error("The local alignment tool could not start.")); });
    child.on("close", (code) => {
      clearTimeout(timeout);
      try { const result: unknown = JSON.parse(output); resolve(result); }
      catch { reject(new Error(code ? "The local tool stopped before confirming its result. Reload the saved alignment before trying again." : (error ? "The alignment tool returned an unreadable result." : "No alignment result was returned."))); }
    });
    child.stdin.on("error", () => {});
    child.stdin.end(body === undefined ? "" : JSON.stringify(body));
  });
}

/** Single range support keeps long seeks local without loading the entire video. */
export async function serveAlignmentFile(request: Request, id: string, resource: "packet" | "video"): Promise<Response> {
  const root = clipRoot(id);
  const name = resource === "packet" ? "packet.json" : sources.find(s => s.id === id)!.video;
  const file = path.join(root, name);
  let size: number;
  try { size = (await fs.promises.stat(file)).size; }
  catch { return Response.json({error: "This review clip has not been prepared yet."}, {status: 404}); }
  const headers = new Headers({ "Content-Type": resource === "packet" ? "application/json" : "video/webm", "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff" });
  let start = 0, end = size-1, status = 200;
  if (resource === "video") {
    headers.set("Accept-Ranges", "bytes");
    const range = request.headers.get("range");
    if (range) {
      const match = /^bytes=(\d*)-(\d*)$/.exec(range);
      if (!match || (!match[1] && !match[2])) return new Response(null,{status:416,headers:{"Content-Range":`bytes */${size}`}});
      if (!match[1]) start = Math.max(0,size-Number(match[2]));
      else { start = Number(match[1]); if(match[2]) end = Math.min(size-1,Number(match[2])); }
      if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start > end || start >= size) return new Response(null,{status:416,headers:{"Content-Range":`bytes */${size}`}});
      status = 206;
      headers.set("Content-Range", `bytes ${start}-${end}/${size}`);
    }
  }
  headers.set("Content-Length",String(end-start+1));
  const stream = fs.createReadStream(file,{start,end});
  return new Response(Readable.toWeb(stream) as ReadableStream, {status,headers});
}
