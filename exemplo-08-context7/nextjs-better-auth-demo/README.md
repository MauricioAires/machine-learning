# Demo Better Auth + Next.js

Projeto demo simples com Next.js App Router, TypeScript, Better Auth, GitHub OAuth e SQLite local.

## Requisitos

- Node.js 20+
- npm
- Uma GitHub OAuth App com callback em `http://localhost:3000/api/auth/callback/github`

## Variáveis de ambiente

Crie um arquivo `.env.local` a partir de `.env.example` e preencha:

- `GITHUB_CLIENT_ID`
- `GITHUB_CLIENT_SECRET`
- `BETTER_AUTH_SECRET`
- `BETTER_AUTH_URL`

Obtenha os dados dos Github em: https://github.com/settings/developers

## Como rodar

1. Instale as dependências.
2. Rode a migração do Better Auth para criar o arquivo `better-auth.sqlite` e as tabelas.
3. Inicie o servidor de desenvolvimento.

## Estrutura principal

- `lib/auth.ts`
- `lib/auth-client.ts`
- `auth.ts`
- `app/api/auth/[...all]/route.ts`
- `app/page.tsx`
- `app/login/page.tsx`
- `components/auth-panel.tsx`

## Observações

- O demo usa `new Database("./better-auth.sqlite")` diretamente.
- O botão de login inicia o fluxo OAuth com GitHub.
- O botão de sair encerra a sessão e atualiza a página.
