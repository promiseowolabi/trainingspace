import { LinkButton } from '@/components/Button'

/** 404 — styled as a Python traceback: the route is not in the vocabulary. */
export default function NotFound() {
  return (
    <section className="relative flex min-h-[70dvh] items-center overflow-hidden">
      <div aria-hidden className="absolute inset-0 bg-blueprint opacity-60" />
      <div className="relative mx-auto max-w-xl px-6 py-24 text-center">
        <p className="font-mono text-label uppercase text-danger">404 — route not found</p>
        <h1 className="mt-4 font-mono text-h2 text-text-1">
          KeyError <span className="text-text-3">(token not in vocab)</span>
        </h1>
        <p className="mt-4 font-mono text-body-sm text-text-3">
          this path tokenizes to nothing the model was trained on —{' '}
          <span className="text-danger">loss = ln(V)</span>, the loss of a guess.
        </p>
        <div className="mt-8 flex justify-center">
          <LinkButton to="/" variant="secondary">
            return to safety →
          </LinkButton>
        </div>
      </div>
    </section>
  )
}
