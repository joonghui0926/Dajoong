# Launch copy

All public posts link to the campaign page. Replace `[CAMPAIGN_URL]` only after
the production route and form have passed privacy and conversion testing.

## English founder launch — LinkedIn

Most BIM demos start with a BIM model.

That is the problem.

For many existing buildings and renovation projects, the useful information is
still trapped in floor-plan PDFs. Before coordination, analytics, or digital
twin work can begin, somebody rebuilds the same geometry by hand.

We are opening **The 100 Drawings Without BIM Project**.

Dajoong will select 100 real, permissioned floor-plan sheets and turn them into
review-gated openBIM models. The first 20 teams receive a private conversion, a
colored IFC/GLB handoff, a BIM Build Sheet, and a guided review.

The important part: we will not hide uncertainty. The Build Sheet records what
the compiler created, what remains review-required, and which exact model and
source produced the result.

Nothing becomes public. Nothing becomes training data. Either requires separate
written permission.

Have one representative sheet and a real use case?

Nominate it: [CAMPAIGN_URL]

#openBIM #BIM #VDC #FacilityManagement #AEC

## Korean founder launch

대부분의 BIM 데모는 이미 잘 만들어진 BIM 모델에서 시작합니다.

하지만 기존 건물과 리노베이션 프로젝트의 현실은 다릅니다. 중요한
정보가 아직 PDF 도면에 있고, 협업이나 분석을 시작하기 전에 누군가가
같은 구조를 다시 모델링합니다.

Dajoong이 **The 100 Drawings Without BIM Project**를 시작합니다.

사용 권한이 확인된 실제 평면도 100장을 검토 가능한 openBIM 모델로
전환합니다. 첫 20개 팀에는 비공개 변환, 컬러 IFC/GLB, BIM Build Sheet,
그리고 함께 보는 검수 세션을 제공합니다.

불확실한 부분은 숨기지 않습니다. Build Sheet에는 무엇이 만들어졌고,
무엇이 사람의 검토를 필요로 하며, 어떤 모델과 원본에서 결과가
나왔는지가 기록됩니다.

공개 활용과 학습 데이터 사용은 각각 별도의 서면 동의가 있을 때만
진행합니다.

실제 업무에 필요한 대표 도면 한 장이 있다면 추천해주세요.

[CAMPAIGN_URL]

## Technical proof post

An AI confidence score is not a BIM acceptance test.

Dajoong separates proposal from proof:

- a compact ONNX model reads the drawing on CPU;
- deterministic geometry resolves metric entities and topology;
- a verifier blocks unsupported or contradictory release states;
- Studio shows the highest-risk elements first;
- IFC and GLB retain source and review metadata.

Our public field test will report the unresolved work, not only the clean render.

The 100 Drawings Without BIM Project: [CAMPAIGN_URL]

## Consultant partner post

We are not trying to remove BIM reviewers from the workflow.

We are trying to remove the hours they spend rebuilding the obvious first pass.

For The 100 Drawings Without BIM Project, Dajoong is inviting BIM consultants
to nominate one client-approved legacy plan and co-review the resulting model.
The consultant owns the relationship and final acceptance. Dajoong supplies the
source-linked first pass, the Build Sheet, and the review queue.

Partner details: [CAMPAIGN_URL]

## Weekly case template

Drawing `[NN]/100`

Use case: `[renovation / FM / coordination / estimating]`

What the compiler built:

- `[wall count]` walls
- `[opening count]` openings
- `[room count]` rooms
- `[object count]` supported fixed objects

What it refused to guess:

- `[review category 1]`
- `[review category 2]`

Output: `[IFC/GLB release state]`

The lesson: `[one technical sentence]`

Have a harder drawing? [CAMPAIGN_URL]

## Short video script — 35 seconds

**0–4s** — Show the original plan. `Most BIM software starts after this becomes a model.`

**4–10s** — Show CPU inference and full-sheet overlay. `Dajoong reads the sheet once with a compact private model.`

**10–18s** — Build walls, openings, rooms, and supported objects. `A deterministic compiler turns evidence into metric BIM entities.`

**18–25s** — Split plan/model selection. `Every element stays linked to the source and its review state.`

**25–31s** — Show Build Sheet. `The result includes what passed, what failed, and what still needs a professional.`

**31–35s** — Campaign card. `Nominate one of the 100 Drawings Without BIM.`
