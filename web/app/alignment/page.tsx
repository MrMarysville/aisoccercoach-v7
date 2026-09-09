import { alignmentClips } from "@/lib/alignment-server";
import AlignmentWorkspace from "./AlignmentWorkspace";
export const dynamic = "force-dynamic";
export default function AlignmentPage() {
  return <AlignmentWorkspace clips={alignmentClips()} />;
}
