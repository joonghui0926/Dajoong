import { DajoongLogo } from "./components/DajoongLogo";

const copy = {
  privacy: {
    title: "Privacy policy",
    updated: "Effective August 9, 2026",
    sections: [
      ["Who operates Dajoong", "Dajoong provides the Plan2BIM service and acts as the data controller for individual accounts. For organization projects, Dajoong processes project data under the organization's instructions. Privacy requests can be sent to jjoonghui@gmail.com."],
      ["What we process", "We process account identifiers, uploaded drawings, conversion settings, generated models, corrections, review history, and limited security or reliability logs needed to provide and protect the service."],
      ["Drawing ownership", "You retain ownership of uploaded drawings and generated project data. Project content is used to run the requested conversion and is never used for model training without separate written permission."],
      ["Why we process it", "Account and project data is processed to perform the service, secure accounts, provide support, and meet contractual or legal obligations. Optional analytics remains off until you consent."],
      ["Storage and retention", "Production uses encrypted transport, encrypted private object storage, scoped access, and audit logs. Personal conversion artifacts expire after the configured retention period, which is 90 days by default. Deleted records may remain in encrypted disaster-recovery backups for up to 35 days before automatic expiry."],
      ["Service providers and transfers", "AWS provides private compute, identity, queues, and storage. Cloudflare provides the public web and API edge. They process data only to provide Dajoong under their service terms. Processing may occur in the United States."],
      ["Your choices", "You can export, correct, or delete personal account data from Studio. Organization administrators control organization-owned project access and retention. Depending on your location, you may also object, restrict processing, or request a portable copy by email."],
      ["Children", "Dajoong is a professional construction tool and is not directed to children under 16."],
      ["Contact", "Privacy and data requests can be sent to jjoonghui@gmail.com."],
    ],
  },
  cookies: {
    title: "Cookie policy",
    updated: "Updated August 8, 2026",
    sections: [
      ["Essential storage", "Dajoong stores your cookie choice and may keep an in-progress Studio session in local browser storage. These functions are required for the requested experience."],
      ["Optional analytics", "Analytics is disabled until you allow it. When enabled, it measures product usage and reliability without reading drawing contents."],
      ["Changing your choice", "Open Account and privacy in Studio to switch between Essential only and Allow analytics at any time. Global Privacy Control always forces optional analytics off."],
      ["Contact", "Questions about cookies can be sent to jjoonghui@gmail.com."],
    ],
  },
  terms: {
    title: "Terms of use",
    updated: "Updated August 8, 2026",
    sections: [
      ["Review responsibility", "Generated BIM is a review-gated digital model. A qualified project professional must approve dimensions, assemblies, and coordination before construction use."],
      ["Permitted use", "You may use Dajoong for projects you own or are authorized to process. Uploaded content must respect third-party rights and project confidentiality."],
      ["Exports", "IFC and GLB exports preserve source references and review state where supported. Downstream software may interpret some metadata differently."],
      ["Availability and warranties", "The service is provided with review controls, but uninterrupted availability and error-free construction output are not guaranteed. Use of generated models for construction remains subject to professional review."],
      ["Suspension and termination", "Access may be suspended for security risk, unlawful use, or material breach. You may stop using the service and delete a personal account at any time."],
      ["Contact", "Commercial and legal questions can be sent to jjoonghui@gmail.com."],
    ],
  },
  support: {
    title: "Support",
    updated: "Available for Dajoong web and mobile",
    sections: [
      ["Product support", "For sign-in, conversion, export, billing, or data questions, email jjoonghui@gmail.com. Include the job ID shown in Studio when reporting a conversion issue, but do not attach confidential drawings unless requested through an approved support channel."],
      ["Privacy support", "Account access, export, correction, and deletion requests can be sent to the same address. We may verify account ownership before acting on a request."],
      ["Security reports", "Send suspected security issues privately to jjoonghui@gmail.com. Do not publish customer drawings, credentials, or exploit details."],
    ],
  },
  accountDeletion: {
    title: "Account deletion",
    updated: "Self-service deletion is available in Dajoong Studio",
    sections: [
      ["Delete inside the app", "Sign in, open Account and privacy, type DELETE, and confirm. Dajoong removes the personal sign-in and personal conversion jobs associated with that identity."],
      ["Delete without the app", "If you cannot access the account, email jjoonghui@gmail.com from the account address with the subject Account deletion. We will verify ownership before deletion."],
      ["What may remain", "Organization-owned construction records may remain under the organization's contract and retention policy. Encrypted disaster-recovery backups can retain deleted records for up to 35 days before automatic expiry."],
    ],
  },
} as const;

export function Legal({ page }: { page: keyof typeof copy }) {
  const content = copy[page];
  return (
    <main className="legal-page">
      <header className="legal-nav"><a href="/"><DajoongLogo /></a><a href="/studio">Open Studio</a></header>
      <article>
        <p className="section-kicker">LEGAL</p>
        <h1>{content.title}</h1>
        <p className="legal-updated">{content.updated}</p>
        {content.sections.map(([title, body]) => <section key={title}><h2>{title}</h2><p>{body}</p></section>)}
        {page === "support" ? <p className="legal-action"><a href="mailto:jjoonghui@gmail.com?subject=Dajoong%20support">Email support</a></p> : null}
        {page === "accountDeletion" ? <p className="legal-action"><a href="/studio">Open Studio account settings</a><a href="mailto:jjoonghui@gmail.com?subject=Account%20deletion">Request by email</a></p> : null}
      </article>
    </main>
  );
}
