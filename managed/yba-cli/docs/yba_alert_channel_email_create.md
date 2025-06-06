## yba alert channel email create

Create a email alert channel in YugabyteDB Anywhere

### Synopsis

Create a email alert channel in YugabyteDB Anywhere

```
yba alert channel email create [flags]
```

### Examples

```
yba alert channel email create --name <alert-channel-name> \
  --use-default-recipients --use-default-smtp-settings
```

### Options

```
      --use-default-rececipients    [Optional] Use default recipients for alert channel. (default false)
      --recipients stringArray      [Optional] Recipients for alert channel. Can be provided as separate flags or as comma-separated values. Required when use-default-recipients is false
      --use-default-smtp-settings   [Optional] Use default SMTP settings for alert channel. Values of smtp-server, smtp-port, email-from, smtp-username, smtp-password, use-ssl, use-tls are used if false. (default false)
      --smtp-server string          [Optional] SMTP server for alert channel. Required when use-default-smtp-settings is false
      --smtp-port int               [Optional] SMTP port for alert channel. Required when use-default-smtp-settings is false (default -1)
      --email-from string           [Optional] SMTP email 'from' address. Required when use-default-smtp-settings is false
      --smtp-username string        [Optional] SMTP username.
      --smtp-password string        [Optional] SMTP password.
      --use-ssl                     [Optional] Use SSL for SMTP connection. (default false)
      --use-tls                     [Optional] Use TLS for SMTP connection. (default false)
  -h, --help                        help for create
```

### Options inherited from parent commands

```
  -a, --apiToken string    YugabyteDB Anywhere api token.
      --ca-cert string     CA certificate file path for secure connection to YugabyteDB Anywhere. Required when the endpoint is https and --insecure is not set.
      --config string      Full path to a specific configuration file for YBA CLI. If provided, this takes precedence over the directory specified via --directory, and the generated files are added to the same path. If not provided, the CLI will look for '.yba-cli.yaml' in the directory specified by --directory. Defaults to '$HOME/.yba-cli/.yba-cli.yaml'.
      --debug              Use debug mode, same as --logLevel debug.
      --directory string   Directory containing YBA CLI configuration and generated files. If specified, the CLI will look for a configuration file named '.yba-cli.yaml' in this directory. Defaults to '$HOME/.yba-cli/'.
      --disable-color      Disable colors in output. (default false)
  -H, --host string        YugabyteDB Anywhere Host (default "http://localhost:9000")
      --insecure           Allow insecure connections to YugabyteDB Anywhere. Value ignored for http endpoints. Defaults to false for https.
  -l, --logLevel string    Select the desired log level format. Allowed values: debug, info, warn, error, fatal. (default "info")
  -n, --name string        [Optional] The name of the alert channel for the operation. Use single quotes ('') to provide values with special characters. Required for create, update, describe, delete.
  -o, --output string      Select the desired output format. Allowed values: table, json, pretty. (default "table")
      --timeout duration   Wait command timeout, example: 5m, 1h. (default 168h0m0s)
      --wait               Wait until the task is completed, otherwise it will exit immediately. (default true)
```

### SEE ALSO

* [yba alert channel email](yba_alert_channel_email.md)	 - Manage YugabyteDB Anywhere email alert notification channels

