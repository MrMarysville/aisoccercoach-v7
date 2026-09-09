import { runAlignment, serveAlignmentFile } from "@/lib/alignment-server";
import { allowsAlignmentWrite } from "@/lib/alignment-origin";
export const runtime = "nodejs";
type Context = { params: Promise<{id:string;resource:string}> };
function resultResponse(result: unknown): Response {
  const value = result as {error?:string;code?:string};
  return Response.json(value, {status: value.error ? (String(value.code).toLowerCase().includes("conflict") || String(value.code).toLowerCase().includes("stale") ? 409 : 422) : 200, headers:{"Cache-Control":"no-store"}});
}
export async function GET(request: Request, context: Context) {
  const {id,resource} = await context.params;
  try {
    if(resource === "packet" || resource === "video") return await serveAlignmentFile(request,id,resource);
    if(resource === "state") return resultResponse(await runAlignment("state",id));
    return Response.json({error:"Unknown review resource."},{status:404});
  } catch (error) { return Response.json({error: error instanceof Error ? error.message : "The clip could not be loaded."},{status:400}); }
}
export async function POST(request: Request, context: Context) {
  // Local browser writes must originate from this application.
  if (!allowsAlignmentWrite(request)) return Response.json({error:"Open this editor from the local application."},{status:403});
  const {id,resource} = await context.params;
  if(resource !== "state" && resource !== "locate") return Response.json({error:"Unknown review action."},{status:404});
  try {
    const text = await request.text();
    if(text.length > 250000) return Response.json({error:"Too many adjustments in one save."},{status:413});
    return resultResponse(await runAlignment(resource === "state" ? "save" : "locate",id,JSON.parse(text)));
  } catch(error) { return Response.json({error:error instanceof Error ? error.message : "The adjustment could not be saved."},{status:400}); }
}
