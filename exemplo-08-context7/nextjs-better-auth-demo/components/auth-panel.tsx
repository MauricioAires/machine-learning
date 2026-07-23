"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { authClient } from "@/lib/auth-client";

type SessionUser = {
  name?: string | null;
  email?: string | null;
};

type AuthPanelProps = {
  user: SessionUser | null;
};

export function AuthPanel({ user }: AuthPanelProps) {
  const router = useRouter();
  const [pending, setPending] = useState(false);
  const loggedLabel = user?.name ?? user?.email ?? "usuário";

  const handleSignIn = async () => {
    setPending(true);
    try {
      const result = await authClient.signIn.social({ provider: "github" });

      if (result.data?.url) {
        window.location.href = result.data.url;
      }
    } finally {
      setPending(false);
    }
  };

  const handleSignOut = async () => {
    setPending(true);
    try {
      await authClient.signOut();
      router.refresh();
    } finally {
      setPending(false);
    }
  };

  return (
    <div className="rounded-[2rem] border border-white/60 bg-white/80 p-6 shadow-glow backdrop-blur-xl sm:p-8">
      {user ? (
        <div className="space-y-5">
          <div>
            <p className="text-sm font-medium uppercase tracking-[0.25em] text-slate-500">Sessão ativa</p>
            <h2 className="mt-2 text-2xl font-semibold text-slate-950">Logado como {loggedLabel}</h2>
            <p className="mt-2 text-sm text-slate-600">
              Nome: {user.name ?? "sem nome"} · Email: {user.email ?? "sem email"}
            </p>
          </div>

          <button
            type="button"
            onClick={handleSignOut}
            disabled={pending}
            className="inline-flex items-center justify-center rounded-full bg-slate-950 px-5 py-3 text-sm font-semibold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {pending ? "Saindo..." : "Sair"}
          </button>
        </div>
      ) : (
        <div className="space-y-5">
          <div>
            <p className="text-sm font-medium uppercase tracking-[0.25em] text-slate-500">Acesso demo</p>
            <h2 className="mt-2 text-2xl font-semibold text-slate-950">Você não está logado</h2>
            <p className="mt-2 text-sm text-slate-600">Entre com sua conta GitHub para criar ou acessar a sessão.</p>
          </div>

          <button
            type="button"
            onClick={handleSignIn}
            disabled={pending}
            className="inline-flex items-center gap-3 rounded-full bg-slate-950 px-5 py-3 text-sm font-semibold text-white transition hover:-translate-y-0.5 hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-60"
          >
            <GitHubIcon />
            {pending ? "Entrando..." : "Entrar com GitHub"}
          </button>
        </div>
      )}
    </div>
  );
}

function GitHubIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" className="h-5 w-5 fill-current">
      <path d="M12 2C6.48 2 2 6.63 2 12.35c0 4.58 2.87 8.47 6.84 9.84.5.1.68-.22.68-.48v-1.7c-2.79.63-3.38-1.38-3.38-1.38-.46-1.22-1.13-1.54-1.13-1.54-.93-.65.07-.64.07-.64 1.03.07 1.57 1.09 1.57 1.09.92 1.63 2.42 1.16 3.01.89.09-.68.36-1.16.65-1.43-2.22-.26-4.55-1.14-4.55-5.06 0-1.12.38-2.03 1-2.75-.1-.26-.43-1.29.1-2.69 0 0 .84-.28 2.75 1.05A9.25 9.25 0 0 1 12 7.98c.85 0 1.72.12 2.53.35 1.91-1.33 2.75-1.05 2.75-1.05.53 1.4.2 2.43.1 2.69.62.72 1 1.63 1 2.75 0 3.93-2.33 4.8-4.56 5.05.37.35.71 1.04.71 2.1v3.11c0 .26.18.59.69.48A10.14 10.14 0 0 0 22 12.35C22 6.63 17.52 2 12 2Z" />
    </svg>
  );
}
