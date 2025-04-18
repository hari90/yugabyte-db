## yba alert channel email

Manage YugabyteDB Anywhere email alert notification channels

### Synopsis

Manage YugabyteDB Anywhere email alert notification channels 

```
yba alert channel email [flags]
```

### Options

```
  -n, --name string   [Optional] The name of the alert channel for the operation. Use single quotes ('') to provide values with special characters. Required for create, update, describe, delete.
  -h, --help          help for email
```

### Options inherited from parent commands

```
  -a, --apiToken string    YugabyteDB Anywhere api token.
      --config string      Full path to a specific configuration file for YBA CLI. If provided, this takes precedence over the directory specified via --directory, and the generated files are added to the same path. If not provided, the CLI will look for '.yba-cli.yaml' in the directory specified by --directory. Defaults to '$HOME/.yba-cli/.yba-cli.yaml'.
      --debug              Use debug mode, same as --logLevel debug.
      --directory string   Directory containing YBA CLI configuration and generated files. If specified, the CLI will look for a configuration file named '.yba-cli.yaml' in this directory. Defaults to '$HOME/.yba-cli/'.
      --disable-color      Disable colors in output. (default false)
  -H, --host string        YugabyteDB Anywhere Host (default "http://localhost:9000")
  -l, --logLevel string    Select the desired log level format. Allowed values: debug, info, warn, error, fatal. (default "info")
  -o, --output string      Select the desired output format. Allowed values: table, json, pretty. (default "table")
      --timeout duration   Wait command timeout, example: 5m, 1h. (default 168h0m0s)
      --wait               Wait until the task is completed, otherwise it will exit immediately. (default true)
```

### SEE ALSO

* [yba alert channel](yba_alert_channel.md)	 - Manage YugabyteDB Anywhere alert notification channels
* [yba alert channel email create](yba_alert_channel_email_create.md)	 - Create a email alert channel in YugabyteDB Anywhere
* [yba alert channel email delete](yba_alert_channel_email_delete.md)	 - Delete YugabyteDB Anywhere email alert channel
* [yba alert channel email describe](yba_alert_channel_email_describe.md)	 - Describe a YugabyteDB Anywhere email alert channel
* [yba alert channel email list](yba_alert_channel_email_list.md)	 - List YugabyteDB Anywhere email alert channels
* [yba alert channel email update](yba_alert_channel_email_update.md)	 - Update a YugabyteDB Anywhere Email alert channel

