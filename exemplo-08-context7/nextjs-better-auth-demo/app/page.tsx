import { headers } from "next/headers";
import Link from "next/link";
import { AuthPanel } from "@/components/auth-panel";
import { auth } from "@/lib/auth";

type SessionUser = {
  name?: string | null;
  email?: string | null;
};

export default async function HomePage() {
  const session = (await auth.api.getSession({ headers: await headers() })) as { user?: SessionUser } | null;

  return (
    <section className="grid w-full gap-6 lg:grid-cols-[1.1fr_0.9fr] lg:gap-10">
      <div className="flex flex-col justify-center gap-6">
        <div className="inline-flex w-fit items-center rounded-full border border-sky-200 bg-white/70 px-4 py-2 text-sm font-medium text-sky-900 shadow-sm backdrop-blur">Next.js + Better Auth + GitHub + SQLite</div>

        <div className="space-y-4">
          <h1 className="max-w-xl text-5xl font-semibold tracking-tight text-slate-950 sm:text-6xl">Hello World com login social e sessão local.</h1>
          <p className="max-w-2xl text-lg leading-8 text-slate-600">Demo enxuto para autenticação com GitHub usando Better Auth, persistindo usuários e sessões em um arquivo SQLite local.</p>
        </div>

        <div className="flex flex-wrap gap-3 text-sm text-slate-600">
          <span className="rounded-full border border-slate-200 bg-white/80 px-4 py-2">App Router</span>
          <span className="rounded-full border border-slate-200 bg-white/80 px-4 py-2">TypeScript</span>
          <span className="rounded-full border border-slate-200 bg-white/80 px-4 py-2">Tailwind CSS</span>
          <span className="rounded-full border border-slate-200 bg-white/80 px-4 py-2">better-sqlite3</span>
        </div>

        <div className="text-sm text-slate-600">
          <Link href="/login" className="font-semibold text-slate-950 underline decoration-sky-400 decoration-2 underline-offset-4">
            Ir para a página de login
          </Link>
        </div>
      </div>

      <div className="flex items-center justify-center">
        <div className="w-full max-w-xl">
          <AuthPanel user={session?.user ?? null} />
        </div>
      </div>
    </section>
  );
}
