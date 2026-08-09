import { DajoongLogo } from "./components/DajoongLogo";

const copy = {
  privacy: {
    title: "Privacy policy",
    updated: "Updated August 8, 2026",
    sections: [
      ["What we process", "Dajoong processes drawings, conversion settings, model corrections, account details, and technical logs needed to provide the service."],
      ["Drawing ownership", "You retain ownership of uploaded drawings and generated project data. Project content is used to run the requested conversion and is never used for model training without separate written permission."],
      ["Storage and security", "Production deployments use encrypted transport, encrypted object storage, scoped access, audit logs, and configurable retention. Local Studio sessions remain in the browser until cleared."],
      ["Your choices", "You can request export, correction, or deletion of account and project data. Organization administrators control project access and retention."],
      ["Contact", "Privacy and data requests can be sent to jjoonghui@gmail.com."],
    ],
  },
  cookies: {
    title: "Cookie policy",
    updated: "Updated August 8, 2026",
    sections: [
      ["Essential storage", "Dajoong stores your cookie choice and may keep an in-progress Studio session in local browser storage. These functions are required for the requested experience."],
      ["Optional analytics", "Analytics is disabled until you allow it. When enabled, it measures product usage and reliability without reading drawing contents."],
      ["Changing your choice", "Clear site data in your browser to reset the choice. A production account settings page can also expose the same control."],
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
      ["Contact", "Commercial and legal questions can be sent to jjoonghui@gmail.com."],
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
      </article>
    </main>
  );
}
