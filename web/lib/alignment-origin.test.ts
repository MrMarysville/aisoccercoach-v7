import { describe, expect, it } from "vitest";
import { allowsAlignmentWrite } from "./alignment-origin";
describe("local alignment write origin",()=>{
  const request=(headers:Record<string,string>)=>new Request("http://localhost:3100/api/alignment/clip/state",{method:"POST",headers:{host:"127.0.0.1:3100",...headers}});
  it("uses the incoming host when Next normalizes the internal request URL",()=>{
    expect(allowsAlignmentWrite(request({origin:"http://127.0.0.1:3100"}))).toBe(true);
    expect(allowsAlignmentWrite(request({host:"localhost:3100",origin:"http://localhost:3100"}))).toBe(true);
    expect(allowsAlignmentWrite(request({}))).toBe(true);
  });
  it("rejects foreign origins, ports, cross-site requests and nonlocal hosts",()=>{
    const blocked:Record<string,string>[]=[
      {origin:"http://untrusted.invalid"},
      {origin:"http://127.0.0.1:3000"},
      {origin:"null"},
      {origin:"http://127.0.0.1:3100","sec-fetch-site":"cross-site"},
      {host:"untrusted.invalid",origin:"http://untrusted.invalid"},
      {host:"attacker@127.0.0.1:3100",origin:"http://127.0.0.1:3100"},
    ];
    for(const headers of blocked)expect(allowsAlignmentWrite(request(headers))).toBe(false);
  });
});
