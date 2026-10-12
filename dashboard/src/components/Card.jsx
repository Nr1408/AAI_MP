export function Card({ children, className = '' }) {
  return (
    <div className={`bg-slate-900/80 border border-slate-700/60 rounded-xl p-4 ${className}`}>
      {children}
    </div>
  )
}

export function CardTitle({ children, icon }) {
  return (
    <div className="flex items-center gap-2 mb-3">
      {icon && <span className="text-indigo-400">{icon}</span>}
      <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">{children}</h3>
    </div>
  )
}
