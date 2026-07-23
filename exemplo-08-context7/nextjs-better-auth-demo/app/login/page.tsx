import { headers } from "next/headers";
import Link from "next/link";
import { AuthPanel } from "@/components/auth-panel";
import { auth } from "@/lib/auth";

type SessionUser = {
  name?: string | null;
  email?: string | null;
};

export default async function LoginPage() {
  const session = (await auth.api.getSession({ headers: await headers() })) as { user?: SessionUser } | null;

  return (
    <section className="mx-auto grid w-full max-w-3xl gap-6">
      <div className="space-y-3 text-center">
        <p className="text-sm font-medium uppercase tracking-[0.28em] text-slate-500">Entrar</p>
        <h1 className="text-4xl font-semibold tracking-tight text-slate-950">Login simples com GitHub</h1>
        <p className="mx-auto max-w-2xl text-base leading-7 text-slate-600">Use a conta GitHub para criar a sessão e volte para a home para ver o estado autenticado.</p>
      </div>

      <AuthPanel user={session?.user ?? null} />

      <div className="text-center text-sm text-slate-600">
        <Link href="/" className="font-semibold text-slate-950 underline decoration-sky-400 decoration-2 underline-offset-4">
          Voltar para a home
        </Link>
      </div>
    </section>
  );
}
