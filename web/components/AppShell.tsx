import type { ReactNode } from "react";
import Link from "next/link";
import styles from "./AppShell.module.css";

/** Top bar plus the one scrolling region. Every screen fits the viewport; panels scroll, the page never does. */
export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className={styles.shell}>
      <header className={styles.bar}>
        <Link href="/" className={styles.brand} aria-label="AI Soccer Coach, field alignment">
          <span className={styles.mark} aria-hidden />
          <span className={`display ${styles.name}`}>AI Soccer Coach</span>
        </Link>
        <nav className={styles.nav} aria-label="Sections">
          <Link href="/alignment">Align field</Link>
        </nav>
        <span className={`mono ${styles.meta}`}>local · this computer only</span>
      </header>
      <main className={styles.main}>{children}</main>
    </div>
  );
}
