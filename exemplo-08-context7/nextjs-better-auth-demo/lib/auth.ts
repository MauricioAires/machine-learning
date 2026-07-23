import { betterAuth } from "better-auth";
import Database from "better-sqlite3";

const database = new Database("./better-auth.sqlite");
const githubClientId = process.env.GITHUB_CLIENT_ID;
const githubClientSecret = process.env.GITHUB_CLIENT_SECRET;

export const auth = betterAuth({
  database,
  secret: process.env.BETTER_AUTH_SECRET ?? "dev-secret-change-me-use-a-long-random-string-for-local-demo",
  baseURL: process.env.BETTER_AUTH_URL ?? "http://localhost:3000",
  trustedOrigins: [process.env.BETTER_AUTH_URL ?? "http://localhost:3000"],
  ...(githubClientId && githubClientSecret
    ? {
        socialProviders: {
          github: {
            clientId: githubClientId,
            clientSecret: githubClientSecret,
          },
        },
      }
    : {}),
});
