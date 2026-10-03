import { useEffect, useState } from 'react'
import { ArrowRight, CheckCircle2, LockKeyhole, Trash2 } from 'lucide-react'
import { researchApi, type LearningSession, type PolicyEvaluation } from './research-api'
import type { EvidenceClaim } from './types'
import './workbench.css'

const SESSION_KEY = 'sca3-learning-session-v3'
const percent = (p: number) => `${Math.round(p * 100)}%`

export function AssessmentStudio() {
  const [session, setSession] = useState<LearningSession | null>(null)
  const [selected, setSelected] = useState<number | null>(null)
  const [feedbackVisible, setFeedbackVisible] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [claims, setClaims] = useState<EvidenceClaim[]>([])
  const [evaluation, setEvaluation] = useState<PolicyEvaluation | null>(null)
  useEffect(() => {
    let active = true
    const token = sessionStorage.getItem(SESSION_KEY)
    if (token) researchApi<LearningSession>(`/api/learning/sessions/${token}`).then(value => { if (active) setSession(value) }).catch(reason => { if (active) setError(`Could not restore practice: ${reason.message}. You can start a new session.`) })
    researchApi<{ claims: EvidenceClaim[] }>('/api/evidence').then(value => { if (active) setClaims(value.claims) }).catch(() => {})
    researchApi<PolicyEvaluation>('/api/learning/policy-evaluation').then(value => { if (active && value.status === 'available') setEvaluation(value) }).catch(() => {})
    return () => { active = false }
  }, [])
  async function start() {
    setBusy(true); setError('')
    try { const state = await researchApi<LearningSession>('/api/learning/sessions', 'POST'); sessionStorage.setItem(SESSION_KEY, state.session_id); setSession(state); setFeedbackVisible(false); setSelected(null) }
    catch (reason) { setError(reason instanceof Error ? reason.message : 'Unable to start') }
    finally { setBusy(false) }
  }
  async function answer() {
    if (!session?.next_item || selected === null) return
    setBusy(true); setError('')
    try { const state = await researchApi<LearningSession>('/api/learning/respond', 'POST', { session_id: session.session_id, question_token: session.next_item.question_token, selected_index: selected }); setSession(state); setFeedbackVisible(true); setSelected(null) }
    catch (reason) { setError(reason instanceof Error ? reason.message : 'Response failed') }
    finally { setBusy(false) }
  }
  async function remove() {
    if (!session) return
    setBusy(true)
    try { await researchApi(`/api/learning/sessions/${session.session_id}`, 'DELETE'); sessionStorage.removeItem(SESSION_KEY); setSession(null); setFeedbackVisible(false); setError('') }
    catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not delete session') }
    finally { setBusy(false) }
  }
  const question = session?.next_item
  const feedback = session?.feedback
  const source = claims.find(claim => claim.id === feedback?.evidence_claim_id)
  return <section className="research-workbench assessment-studio" aria-label="Adaptive evidence literacy practice">
    <header className="wb-title"><span className="wb-kicker">06 / LEARNING SCIENCES</span><h1>Understanding can be inspected.</h1><p>Practice interpreting evidence. See why an answer matters, how it changes the learner model, and which prerequisite determines the next question.</p></header>
    {!session ? <article className="wb-card wb-learning-intro"><div><span className="wb-kicker">ADAPTIVE PRACTICE / 6 CONCEPTS</span><h2>Evidence literacy, one decision at a time.</h2><p>A small, source-linked item bank with randomized option order, misconception feedback, prerequisite repair and transparent Bayesian updates.</p><p className="wb-boundary">Starting saves anonymous practice responses locally. Access expires after seven days; expired records are cleared when a new session starts. You can delete your session now. No name, patient history, diagnosis or free text is requested. Learning effectiveness has not been tested with people.</p><button className="primary-button" onClick={start} disabled={busy}>{busy ? 'Starting…' : 'Start local practice'}<ArrowRight size={16} /></button></div><div className="wb-equation">Prior belief<br />↓<br />Response evidence<br />↓<br />Feedback & next step</div></article> : <div className="wb-assessment-grid">
      <aside className="wb-card wb-mastery"><span className="wb-kicker">LEARNER MODEL / NOT A TEST SCORE</span><h2>Concept map</h2>{session.concepts.map(concept => { const unlocked = concept.prerequisites.every(parent => session.mastery[parent] >= .65); const belief = session.mastery[concept.id]; return <div className={`wb-concept ${unlocked ? '' : 'locked'}`} key={concept.id}><div><span>{!unlocked && <LockKeyhole size={12} />}{concept.title}</span><b>{percent(belief)}</b></div><meter min={0} max={1} value={belief} aria-label={`${concept.title} model belief`} /><small>{unlocked ? `Threshold ${percent(concept.mastery_threshold)} · model assumption` : `Prerequisite: ${concept.prerequisites.join(' + ').replaceAll('_', ' ')}`}</small></div> })}<p className="wb-small">{session.completed_items} responses in this session. Repeated questions receive an explicit memory adjustment.</p><button className="text-button wb-delete" onClick={remove} disabled={busy}><Trash2 size={14} />Delete this practice session</button></aside>
      <article className="wb-card wb-question" aria-live="polite" aria-busy={busy}>
        {feedbackVisible && feedback ? <div className="wb-feedback"><span className="wb-kicker">{feedback.correct ? 'REASONING SUPPORTED' : 'REVISIT THE DISTINCTION'}</span><h2>{feedback.correct ? 'That interpretation preserves the boundary.' : 'Here is the important distinction.'}</h2><p className="wb-question-recap">{feedback.prompt}</p><div className="wb-feedback-answer"><strong>Your response</strong><p>{feedback.selected_option}</p><strong>Supported interpretation</strong><p>{feedback.correct_option}</p></div><p>{feedback.rationale}</p>{feedback.misconception && <p className="wb-small">Reasoning pattern: {feedback.misconception.replaceAll('_', ' ')}</p>}<div className="wb-update-path"><div><span>Prior</span><strong>{percent(feedback.update.prior)}</strong></div><ArrowRight size={16} /><div><span>After response</span><strong>{percent(feedback.update.posterior_after_observation)}</strong></div><ArrowRight size={16} /><div><span>After learning assumption</span><strong>{percent(feedback.update.posterior_after_learning)}</strong></div></div>{feedback.review_assumption && <p className="wb-small">{feedback.review_assumption}</p>}{source && <a className="wb-source-link" href={source.source_url} target="_blank" rel="noreferrer">Read the linked evidence: {source.source_title}</a>}<button className="primary-button" onClick={() => setFeedbackVisible(false)}>Continue<ArrowRight size={16} /></button></div> : question ? <form onSubmit={(e) => { e.preventDefault(); void answer() }}><div className="wb-card-heading"><span className="wb-kicker">{question.concept_id.replaceAll('_', ' ')} / {question.id}</span>{question.selection.review && <span className="wb-review-tag">REVIEW / repeated item</span>}</div><fieldset><legend>{question.prompt}</legend><div className="wb-options">{question.options.map((option, index) => <label key={option} className={selected === index ? 'selected' : ''}><input type="radio" name={question.question_token} checked={selected === index} onChange={() => setSelected(index)} /><span>{option}</span></label>)}</div></fieldset><button className="primary-button" type="submit" disabled={selected === null || busy}>{busy ? 'Updating learner model…' : 'Submit interpretation'}<CheckCircle2 size={16} /></button><details className="wb-disclosure"><summary>Why was this question selected?</summary><p>{question.selection.policy}</p><p>Expected information gain: {question.selection.expected_information_gain_bits.toFixed(3)} bits.</p><p className="wb-small">Item selection uses modeled learning need and uncertainty. This does not establish that the policy improves learning.</p></details></form> : <div><span className="wb-kicker">PRACTICE COMPLETE</span><h2>{session.completion === 'practice_limit_reached' ? 'Time to pause and reflect.' : 'The model thresholds have been reached.'}</h2><p>Explain, in your own words, why a cohort average, a simulated treatment effect and an individual prognosis are different claims.</p><p className="wb-boundary">This is a reflection prompt, not a measured transfer outcome. Model thresholds do not certify competence.</p><button className="text-button" onClick={remove} disabled={busy}>Delete session and return</button></div>}
        <p className="wb-small wb-learning-boundary">{session.interpretation_boundary}</p>
      </article>
    </div>}
    {error && <p role="alert" className="wb-error">{error}</p>}
    {evaluation && <article className="wb-card"><span className="wb-kicker">POLICY EVALUATION / SIMULATED LEARNERS ONLY</span><h2>Does the adaptive rule outperform a simple comparator?</h2><div className="wb-stat-grid"><div><span>Adaptive rule</span><strong>{percent(evaluation.adaptive.expected_posttest_accuracy)}</strong><small>Expected synthetic post-test accuracy</small></div><div><span>Simple eligible-item comparator</span><strong>{percent(evaluation.fixed_sequence.expected_posttest_accuracy)}</strong><small>Same prerequisite constraints</small></div><div><span>Paired difference</span><strong>{(100 * evaluation.paired_difference_intervals.expected_posttest_accuracy.mean).toFixed(2)} pp</strong><small>95% MC interval {(100 * evaluation.paired_difference_intervals.expected_posttest_accuracy.low).toFixed(2)} to {(100 * evaluation.paired_difference_intervals.expected_posttest_accuracy.high).toFixed(2)} pp</small></div></div><p>{evaluation.paired_difference_intervals.expected_posttest_accuracy.mean < 0 ? 'The current adaptive heuristic did not outperform the comparator under this simulator. This negative result is retained; no learning-benefit claim is made.' : 'The modeled difference is specific to this simulator. It does not establish a learning benefit in people.'}</p><details className="wb-disclosure"><summary>Inspect the comparison design</summary><p>{evaluation.comparison_design}</p><p>Comparator: {evaluation.comparator_policy}</p><p>{evaluation.learners.toLocaleString()} paired synthetic learners · {evaluation.steps} practice steps · seed {evaluation.seed}</p><p>{evaluation.interpretation_boundary}</p><a className="text-button" href="/api/learning/policy-evaluation" target="_blank" rel="noreferrer">Inspect complete evaluation</a></details></article>}
  </section>
}
