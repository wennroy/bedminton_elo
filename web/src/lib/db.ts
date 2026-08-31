import Database from "better-sqlite3";
import { readFileSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export function getDatabaseUrl(): string {
  return process.env.DATABASE_URL ?? "./badminton.db";
}

export function createDb(url?: string): Database.Database {
  const db = new Database(url ?? getDatabaseUrl());
  db.pragma("journal_mode = WAL");
  const schemaPath = join(__dirname, "schema.sql");
  const schema = readFileSync(schemaPath, "utf-8");
  db.exec(schema);
  return db;
}

let cachedDb: Database.Database | null = null;

export function getDb(): Database.Database {
  if (!cachedDb) {
    cachedDb = createDb();
  }
  return cachedDb;
}

export function closeDb(): void {
  if (cachedDb) {
    cachedDb.close();
    cachedDb = null;
  }
}
