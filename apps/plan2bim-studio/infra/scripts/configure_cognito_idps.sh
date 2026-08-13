#!/usr/bin/env bash
set -euo pipefail

: "${DAJOONG_USER_POOL_ID:?Set DAJOONG_USER_POOL_ID}"
: "${DAJOONG_USER_POOL_CLIENT_ID:?Set DAJOONG_USER_POOL_CLIENT_ID}"

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT
providers=(COGNITO)

upsert_provider() {
  local name="$1" type="$2" details="$3" mapping="$4"
  jq -n \
    --arg pool "$DAJOONG_USER_POOL_ID" \
    --arg name "$name" \
    --arg type "$type" \
    --argjson details "$details" \
    --argjson mapping "$mapping" \
    '{UserPoolId:$pool,ProviderName:$name,ProviderType:$type,ProviderDetails:$details,AttributeMapping:$mapping}' \
    > "$tmp_dir/$name.json"
  if aws cognito-idp describe-identity-provider --user-pool-id "$DAJOONG_USER_POOL_ID" --provider-name "$name" >/dev/null 2>&1; then
    jq 'del(.ProviderType)' "$tmp_dir/$name.json" > "$tmp_dir/$name-update.json"
    aws cognito-idp update-identity-provider --cli-input-json "file://$tmp_dir/$name-update.json" >/dev/null
  else
    aws cognito-idp create-identity-provider --cli-input-json "file://$tmp_dir/$name.json" >/dev/null
  fi
  providers+=("$name")
}

if [[ -n "${GOOGLE_OAUTH_CLIENT_ID:-}" && -n "${GOOGLE_OAUTH_CLIENT_SECRET:-}" ]]; then
  upsert_provider Google Google \
    "$(jq -nc --arg id "$GOOGLE_OAUTH_CLIENT_ID" --arg secret "$GOOGLE_OAUTH_CLIENT_SECRET" '{client_id:$id,client_secret:$secret,authorize_scopes:"openid email profile"}')" \
    '{"email":"email","name":"name","picture":"picture"}'
fi

provider_json="$(printf '%s\n' "${providers[@]}" | jq -R . | jq -s .)"
aws cognito-idp describe-user-pool-client \
  --user-pool-id "$DAJOONG_USER_POOL_ID" \
  --client-id "$DAJOONG_USER_POOL_CLIENT_ID" \
  | jq --argjson providers "$provider_json" '.UserPoolClient | {
      UserPoolId: env.DAJOONG_USER_POOL_ID,
      ClientId: env.DAJOONG_USER_POOL_CLIENT_ID,
      ClientName, RefreshTokenValidity, AccessTokenValidity, IdTokenValidity,
      TokenValidityUnits, ReadAttributes, WriteAttributes, ExplicitAuthFlows,
      CallbackURLs, LogoutURLs, AllowedOAuthFlows, AllowedOAuthScopes,
      AllowedOAuthFlowsUserPoolClient, PreventUserExistenceErrors,
      EnableTokenRevocation, AuthSessionValidity
    } | .SupportedIdentityProviders=$providers | with_entries(select(.value != null))' \
  > "$tmp_dir/client.json"
aws cognito-idp update-user-pool-client --cli-input-json "file://$tmp_dir/client.json" >/dev/null

printf 'DAJOONG_AUTH_PROVIDERS=%s\n' "$(IFS=,; echo "${providers[*]}")"
