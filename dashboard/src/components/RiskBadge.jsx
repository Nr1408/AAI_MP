import { RISK_BG } from '../lib/constants.js'
import { clsx } from '../lib/utils.js'

export function RiskBadge({ risk }) {
  return (
    <span className={clsx(
      'text-xs font-semibold px-2 py-0.5 rounded-full border',
      RISK_BG[risk] || 'bg-slate-700 text-slate-300 border-slate-600'
    )}>
      {risk || '—'}
    </span>
  )
}
