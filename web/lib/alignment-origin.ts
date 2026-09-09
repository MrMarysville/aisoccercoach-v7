/** Next may normalize request.url to localhost; the incoming Host binds browser origin. */
export function allowsAlignmentWrite(request: Request): boolean {
  if(request.headers.get("sec-fetch-site")==="cross-site")return false;
  try {
    const authority=request.headers.get("host") ?? new URL(request.url).host;
    const host=new URL(`http://${authority}`);
    if(!["127.0.0.1","localhost","[::1]"].includes(host.hostname) || host.username || host.password || host.pathname!=="/")return false;
    const origin=request.headers.get("origin");
    if(!origin)return true; // Local command-line clients do not send Origin.
    const sender=new URL(origin);
    return ["http:","https:"].includes(sender.protocol) && sender.origin===origin && sender.host===host.host;
  }catch{return false;}
}
