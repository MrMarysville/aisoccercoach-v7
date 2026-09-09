/** Browser projection mirrors field_adjustment.py; persistence revalidates every drag. */
export type Point = [number, number];
export type Matrix = [Point3, Point3, Point3];
type Point3 = [number, number, number];
export type Adjustment = {
  mode: "local" | "global";
  center_world: Point;
  delta_reference: Point;
  anchor: { frame_index: number; source_pts: number; source_time_base: string; image_sha256: string; from_native: Point; to_native: Point };
};
export type AlignmentFrame = {
  source_pts: number; source_time_base: string; source_seconds: number; image_sha256: string;
  reference_to_native: Matrix; base_pixels: (Point | null)[][]; warnings: string[];
};
export type AlignmentPacket = {
  id: string; title: string; source_sha256: string; native_size: Point; fps: number;
  duration_s: number; source_start_s: number; base_sha256: string;
  markings: { id: string; label: string; world: Point[]; observed_support: boolean[] }[]; frames: AlignmentFrame[];
};
export type AlignmentState = { revision: string; operations: Adjustment[]; confirmed: boolean; updated_at: string | null };
export type AlignmentClip = { id: string; title: string; description: string; ready: boolean };
export type MarkingHit = {marking:number;sample:number;world:Point;pixel:Point};

export function draftStorageKey(packet: AlignmentPacket): string {
  return `field-alignment-draft-v1:${packet.id}:${packet.source_sha256}:${packet.base_sha256}`;
}

/** Tab-local recovery is bound to the exact saved revision; the server still validates saves. */
export function parseDraft(text: string | null, state: AlignmentState): {operations:Adjustment[];confirmed:boolean}|null {
  if(!text || text.length>250000)return null;
  try {
    const d=JSON.parse(text);
    const point=(p:unknown):p is Point=>Array.isArray(p)&&p.length===2&&p.every(n=>typeof n==="number"&&Number.isFinite(n));
    if(d.version!==1 || d.revision!==state.revision || typeof d.confirmed!=="boolean" || !Array.isArray(d.operations) || d.operations.length>100)return null;
    if(!d.operations.every((op:Adjustment)=>op && (op.mode==="local"||op.mode==="global") && point(op.center_world) && point(op.delta_reference) && op.anchor && Number.isSafeInteger(op.anchor.frame_index) && Number.isSafeInteger(op.anchor.source_pts) && typeof op.anchor.source_time_base==="string" && typeof op.anchor.image_sha256==="string" && point(op.anchor.from_native) && point(op.anchor.to_native)))return null;
    return {operations:d.operations,confirmed:d.confirmed};
  }catch{return null;}
}

export function transform(m: Matrix, [x, y]: Point): Point {
  const z = m[2][0] * x + m[2][1] * y + m[2][2];
  return [(m[0][0] * x + m[0][1] * y + m[0][2]) / z, (m[1][0] * x + m[1][1] * y + m[1][2]) / z];
}

export function inverse(m: Matrix): Matrix {
  const [[a,b,c],[d,e,f],[g,h,i]] = m;
  const det = a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g);
  if (!Number.isFinite(det) || Math.abs(det) < 1e-14) throw new Error("The camera mapping is unavailable.");
  return [[(e*i-f*h)/det,(c*h-b*i)/det,(b*f-c*e)/det],[(f*g-d*i)/det,(a*i-c*g)/det,(c*d-a*f)/det],[(d*h-e*g)/det,(b*g-a*h)/det,(a*e-b*d)/det]];
}

export function localWeight(world: Point, center: Point): number {
  const r = Math.min(1, Math.hypot(world[0]-center[0], world[1]-center[1])/20);
  return (1-r)**4 * (4*r+1);
}

export function adjustedPoint(base: Point, world: Point, m: Matrix, inv: Matrix, operations: Adjustment[]): Point {
  if (!operations.length) return base;
  const ref = transform(inv, base);
  for (const op of operations) {
    const w = op.mode === "global" ? 1 : localWeight(world, op.center_world);
    ref[0] += op.delta_reference[0] * w;
    ref[1] += op.delta_reference[1] * w;
  }
  return transform(m, ref);
}

export function makeAdjustment(frame: AlignmentFrame, frameIndex: number, mode: Adjustment["mode"], world: Point, from: Point, to: Point): Adjustment {
  const inv = inverse(frame.reference_to_native);
  const a = transform(inv, from), b = transform(inv, to);
  return { mode, center_world: world, delta_reference: [b[0]-a[0], b[1]-a[1]], anchor: { frame_index: frameIndex, source_pts: frame.source_pts, source_time_base: frame.source_time_base, image_sha256: frame.image_sha256, from_native: from, to_native: to } };
}

/** Keep unsupported extrapolations out of both drawing and mouse selection. */
export function projectMarkings(packet: AlignmentPacket, frame: AlignmentFrame, operations: Adjustment[]): (Point|null)[][] {
  const inv = inverse(frame.reference_to_native);
  return packet.markings.map((mark,mi)=>mark.world.map((world,pi)=>{
    const base=frame.base_pixels[mi]?.[pi];
    if(mark.observed_support?.[pi]!==true || !base)return null;
    const point=adjustedPoint(base,world,frame.reference_to_native,inv,operations);
    return point.every(Number.isFinite) ? point : null;
  }));
}

export function markingPath(points:(Point|null)[]):string {
  let connected=false, path="";
  for(const p of points){
    if(!p || !p.every(Number.isFinite) || Math.abs(p[0])>100000 || Math.abs(p[1])>100000){connected=false;continue;}
    path+=`${connected?"L":"M"}${p[0].toFixed(2)} ${p[1].toFixed(2)} `;connected=true;
  }
  return path;
}

/** Select the drawn segment, retaining an exact, visible sample for save validation. */
export function nearestMarking(packet: AlignmentPacket, projected:(Point|null)[][], point:Point, radius:number):MarkingHit|null {
  let best=radius*radius, hit:MarkingHit|null=null;
  const drawable=(p:Point|null|undefined):p is Point=>!!p && p.every(Number.isFinite) && p.every(n=>Math.abs(n)<=100000);
  const visible=(p:Point)=>p[0]>=0 && p[1]>=0 && p[0]<packet.native_size[0] && p[1]<packet.native_size[1];
  const distance=(p:Point)=>(p[0]-point[0])**2+(p[1]-point[1])**2;
  const consider=(marking:number,sample:number,pixel:Point,d:number)=>{
    if(d<best){best=d;hit={marking,sample,world:packet.markings[marking]!.world[sample]!,pixel};}
  };
  projected.forEach((pixels,mi)=>pixels.forEach((pixel,pi)=>{
    if(!drawable(pixel))return;
    if(visible(pixel))consider(mi,pi,pixel,distance(pixel));
    const previous=pixels[pi-1];
    if(!drawable(previous))return;
    const dx=pixel[0]-previous[0],dy=pixel[1]-previous[1],length=dx*dx+dy*dy;
    const t=length ? Math.max(0,Math.min(1,((point[0]-previous[0])*dx+(point[1]-previous[1])*dy)/length)) : 0;
    const nearest:Point=[previous[0]+t*dx,previous[1]+t*dy];
    if(!visible(nearest))return;
    const sample=visible(previous) && (!visible(pixel) || distance(previous)<distance(pixel)) ? pi-1 : pi;
    const anchor=pixels[sample]!;
    if(visible(anchor))consider(mi,sample,anchor,distance(nearest));
  }));
  return hit;
}

export function dragTarget(from:Point, start:Point, pointer:Point, size:Point):Point {
  return [Math.max(0,Math.min(size[0]-1,from[0]+pointer[0]-start[0])),
          Math.max(0,Math.min(size[1]-1,from[1]+pointer[1]-start[1]))];
}

export function frameAt(time: number, fps: number, count: number): number {
  return Math.max(0, Math.min(count-1, Math.floor(time * fps + 1e-5)));
}

export function clockTime(seconds: number): string {
  const tenths = Math.round(seconds * 10);
  return `${Math.floor(tenths/600)}:${String(Math.floor(tenths/10)%60).padStart(2,"0")}.${tenths%10}`;
}

export function markingName(id: string): string {
  const names:Record<string,string>={touch_far:"Far touchline",touch_near:"Near touchline",halfway:"Halfway line",goal_left:"Left goal line",goal_right:"Right goal line",centre_circle:"Centre circle",pen_arc_left:"Left penalty arc",pen_arc_right:"Right penalty arc"};
  if(names[id])return names[id];
  const box=/^box(18|6)_(left|right)_(front|near|far)$/.exec(id);
  if(box)return `${box[2]==="left"?"Left":"Right"} ${box[1]==="18"?"penalty":"goal"} area · ${box[3]==="front"?"front":`${box[3]} side`}`;
  return "Field marking";
}
