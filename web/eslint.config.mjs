import nextCoreWebVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

// Images are local files served by our own route; next/image optimisation has nothing to add.
const config = [...nextCoreWebVitals, ...nextTs, { ignores: [".next/**", "node_modules/**"] }, { rules: { "@next/next/no-img-element": "off" } }];
export default config;
