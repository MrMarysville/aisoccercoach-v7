import { describe, expect, it } from "vitest";
import { adjustedPoint, dragTarget, frameAt, inverse, localWeight, makeAdjustment, markingPath, nearestMarking, parseDraft, projectMarkings, transform, type AlignmentFrame, type AlignmentPacket, type Matrix, type Point } from "./alignment";
const m:Matrix=[[1.2,.02,170],[.01,1.1,70],[.0001,-.00003,1]];
const frame:AlignmentFrame={source_pts:55803000,source_time_base:"1/90000",source_seconds:620.0333333333,image_sha256:"image",reference_to_native:m,base_pixels:[],warnings:[]};
describe("source-bound field adjustments",()=>{
  it("projects an arbitrary drag to its target and transports it through a pan",()=>{
    const base:Point=[630,440],world:Point=[10,15],to:Point=[635.3,437.6];
    const op=makeAdjustment(frame,0,"local",world,base,to);
    expect(adjustedPoint(base,world,m,inverse(m),[op])[0]).toBeCloseTo(to[0],8);
    expect(adjustedPoint(base,world,m,inverse(m),[op])[1]).toBeCloseTo(to[1],8);
    const pan:Matrix=[[.8,.05,-290],[.01,.9,25],[-.0001,.0001,1]];
    const ref=transform(inverse(m),base), later=transform(pan,ref);
    const expected=transform(pan,[ref[0]+op.delta_reference[0],ref[1]+op.delta_reference[1]]);
    expect(adjustedPoint(later,world,pan,inverse(pan),[op])).toEqual(expected);
    expect(op.anchor.source_pts).toBe(55803000);
    expect(op.anchor.image_sha256).toBe("image");
  });
  it("leaves the original exact with no operations, and tapers local moves to zero",()=>{
    const base:Point=[100,200];
    expect(adjustedPoint(base,[0,0],m,inverse(m),[])).toBe(base);
    expect(localWeight([0,0],[0,0])).toBe(1);
    expect(localWeight([20,0],[0,0])).toBe(0);
    expect(localWeight([100,100],[0,0])).toBe(0);
    const op=makeAdjustment(frame,0,"local",[0,0],base,[110,210]);
    const point=adjustedPoint(base,[30,30],m,inverse(m),[op]);
    expect(point[0]).toBeCloseTo(base[0],10);
    expect(point[1]).toBeCloseTo(base[1],10);
  });
  it("selects the presented 10 fps source frame at boundaries and at the end",()=>{
    expect(frameAt(0,10,351)).toBe(0);
    expect(frameAt(.099,10,351)).toBe(0);
    expect(frameAt(.1,10,351)).toBe(1);
    expect(frameAt(35.1,10,351)).toBe(350);
    expect(frameAt(-1,10,351)).toBe(0);
  });
  it("recovers a tab draft only against its saved revision and rejects malformed geometry",()=>{
    const state={revision:"saved-a",operations:[],confirmed:false,updated_at:null};
    const op=makeAdjustment(frame,0,"local",[0,0],[630,440],[632,443]);
    const draft={version:1,revision:"saved-a",operations:[op],confirmed:false};
    expect(parseDraft(JSON.stringify(draft),state)?.operations).toEqual([op]);
    expect(parseDraft(JSON.stringify({...draft,revision:"saved-b"}),state)).toBeNull();
    expect(parseDraft(JSON.stringify({...draft,operations:[{...op,center_world:[null,0]}]}),state)).toBeNull();
    expect(parseDraft("not json",state)).toBeNull();
  });
});

// Synthetic pole crossing: the offscreen endpoints must never be joined through the video.
it("breaks unsupported line sections while preserving supported coordinates and edits",()=>{
  const packet={markings:[{id:"touch_near",label:"touch_near",world:[[0,0],[1,0],[2,0],[3,0],[4,0]],observed_support:[true,false,false,true,true]}]} as AlignmentPacket;
  const f={...frame,base_pixels:[[[100,200],[8069,-10788],[-19484,37878],[200,300],[210,310]]] } as AlignmentFrame;
  const points=projectMarkings(packet,f,[])[0]!;
  expect(points).toEqual([[100,200],null,null,[200,300],[210,310]]);
  expect(markingPath(points)).toBe("M100.00 200.00 M200.00 300.00 L210.00 310.00 ");
  const op=makeAdjustment(f,0,"global",[0,0],[100,200],[105,203]);
  const moved=projectMarkings(packet,f,[op])[0]!;
  expect(moved[0]![0]).toBeCloseTo(105,8);
  expect(moved[0]![1]).toBeCloseTo(203,8);
  expect(moved[1]).toBeNull();
  expect(moved[2]).toBeNull();
  const missing={...packet,markings:[{...packet.markings[0]!,observed_support:undefined}]} as unknown as AlignmentPacket;
  expect(projectMarkings(missing,f,[])[0]).toEqual([null,null,null,null,null]);
});


it("grabs segment middles using an exact visible anchor and does not bridge gaps",()=>{
  const packet={native_size:[1920,1080],markings:[{world:[[0,0],[1,0],[2,0],[3,0]]}]} as AlignmentPacket;
  const points:(Point|null)[][]=[[[100,200],[310,200],null,[600,200]]];
  const hit=nearestMarking(packet,points,[205,205],20)!;
  expect(hit).not.toBeNull();
  expect(hit.pixel).toBe(points[0]![hit.sample]);
  expect(hit.world).toBe(packet.markings[0]!.world[hit.sample]);
  expect(nearestMarking(packet,points,[450,200],20)).toBeNull();
  expect(nearestMarking(packet,points,[205,225],20)).toBeNull();
  const clipped=nearestMarking(packet,[[[-100,200],[100,200]]],[5,200],20)!;
  expect(clipped.sample).toBe(1);
});

it("keeps captured drags within every image edge without changing interior movement",()=>{
  expect(dragTarget([100,100],[105,105],[120,125],[1920,1080])).toEqual([115,120]);
  expect(dragTarget([100,100],[105,105],[-30,-50],[1920,1080])).toEqual([0,0]);
  expect(dragTarget([100,100],[105,105],[2000,1200],[1920,1080])).toEqual([1919,1079]);
});
