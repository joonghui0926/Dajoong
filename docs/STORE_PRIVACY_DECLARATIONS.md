# Store privacy declarations

These answers describe the current production code. Re-review them before each
release if authentication, analytics, uploads, customer support, advertising,
or third-party SDKs change.

## Apple App Privacy

**Tracking:** No.

**Data used for third-party advertising:** No.
**Data linked to the user:** Yes, only for app functionality and account
management.

Declare:

| Apple category | Linked | Tracking | Purpose |
| --- | --- | --- | --- |
| Name | Yes | No | App functionality |
| Email Address | Yes | No | App functionality, account management |
| User ID | Yes | No | App functionality, account management |
| Other User Content | Yes | No | Drawing conversion, editing, review, export |

`PrivacyInfo.xcprivacy` makes the same no-tracking declaration. Project drawings,
photos embedded in submitted documents, PlanGraph corrections, and review notes
are covered by Other User Content. Do not mark data as collected for analytics
unless an actual analytics service is added and consent-gated.

## Google Play Data safety

**Does the app collect or share required user data types?** Collects; does not
sell or share for advertising. AWS and Cloudflare operate as contracted service
providers processing data on behalf of Dajoong.

Declare as collected:

| Google category | Required | Purpose |
| --- | --- | --- |
| Name | Optional | Account management, app functionality |
| Email address | Required | Account management, authentication |
| User IDs | Required | Account management, security |
| Files and documents | Optional | Drawing conversion, app functionality |
| Photos | Optional | User-selected drawing/evidence input |
| Other user-generated content | Optional | Corrections, review notes, model patches |

For every collected category: data is encrypted in transit, is not used for
advertising, and is not sold. Users can request deletion from inside the app or
through `https://studio.dajoongbim.com/account-deletion`.

Answer **No** for precise/approximate location, contacts, financial information,
health, messages, audio, browsing history, advertising identifiers, and device
tracking because the current app neither requests those permissions nor ships
an SDK that collects them. A user-selected document may contain such information;
it is processed as user content rather than independently inferred profile data.

## Cookies and local storage

The native WebView and web app use essential local storage for consent,
authentication transaction state, and recoverable editor state. Optional
analytics stays disabled by default and Global Privacy Control forces the
essential-only choice. Authentication tokens are not written to the marketing
site. The system browser handles OAuth and PKCE.

## Public contacts

- Privacy: `https://studio.dajoongbim.com/privacy`
- Cookies: `https://studio.dajoongbim.com/cookies`
- Support: `https://studio.dajoongbim.com/support`
- Account deletion: `https://studio.dajoongbim.com/account-deletion`
- Contact: `jjoonghui@gmail.com`
